from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import video_script_reconstructor.review_batch as review_batch


def _project(root: Path, name: str) -> Path:
    project = root / name
    state = project / ".state"
    state.mkdir(parents=True)
    (state / "canonical-project.json").write_text(
        json.dumps({"manifest": {"run_id": name}, "visual_events": []}),
        encoding="utf-8",
    )
    return project


def test_discover_review_projects_is_deterministic_and_prunes_project_tree(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    first = _project(corpus, "b-project")
    second = _project(corpus, "a-project")
    (first / "evidence" / "large").mkdir(parents=True)
    (first / "evidence" / "large" / "ignored.bin").write_bytes(b"x")
    (corpus / "not-a-project").mkdir()

    assert review_batch.discover_review_projects(corpus) == (second, first)


def test_estimate_review_bundle_bytes_is_bounded_per_packet() -> None:
    assert review_batch.estimate_review_bundle_bytes(2) == 2 * 4 * 1024**2 + 1024**2
    with pytest.raises(ValueError):
        review_batch.estimate_review_bundle_bytes(0)


def test_tree_bytes_counts_nested_regular_files_without_following_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle-tree"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "request.json").write_bytes(b"request")
    (nested / "response.json").write_bytes(b"response")

    expected = len(b"request") + len(b"response")
    monkeypatch.setattr(
        review_batch.os,
        "walk",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("review bundle tree inventory must use scandir")
        ),
    )
    assert review_batch._tree_bytes(root) == expected

    link = root / "linked-tree"
    try:
        link.symlink_to(nested, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")
    assert review_batch._tree_bytes(root) == expected


def test_create_review_bundles_dry_run_skips_empty_and_never_copies_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    empty = _project(corpus, "empty")
    pending = _project(corpus, "pending")
    evidence = pending / "evidence"
    evidence.mkdir()
    (evidence / "frame.png").write_bytes(b"pixels")
    pending_counts = {empty: 0, pending: 2}
    monkeypatch.setattr(review_batch, "pending_packet_count", pending_counts.__getitem__)
    monkeypatch.setattr(
        review_batch.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(
            total=100 * 1024**2, used=1, free=99 * 1024**2
        ),
    )
    called = False

    def fail_create(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("dry run must not create a bundle")

    monkeypatch.setattr(review_batch, "create_review_bundle", fail_create)
    summary = review_batch.create_review_bundles(
        corpus,
        output_root=tmp_path / "handoff",
        min_free_bytes=1,
        dry_run=True,
    )

    assert called is False
    assert summary["created_count"] == 0
    assert [item["status"] for item in summary["projects"]] == [
        "skipped_no_pending_packets",
        "planned",
    ]
    assert list((tmp_path / "handoff").rglob("*.png")) == []


def test_create_review_bundles_blocks_before_writing_when_reserve_is_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    project = _project(corpus, "pending")
    monkeypatch.setattr(review_batch, "pending_packet_count", lambda _path: 1)
    monkeypatch.setattr(
        review_batch.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(total=100, used=99, free=1),
    )
    monkeypatch.setattr(
        review_batch,
        "create_review_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not write")),
    )

    summary = review_batch.create_review_bundles(corpus, min_free_bytes=2)

    assert summary["blocked"] is True
    assert summary["projects"][0]["status"] == "storage_blocked"
    assert not (project / ".state" / "vision").exists()


def test_create_review_bundles_blocks_when_budget_cannot_hold_one_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _project(corpus, "pending")
    monkeypatch.setattr(review_batch, "pending_packet_count", lambda _path: 1)
    monkeypatch.setattr(
        review_batch,
        "create_review_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not write")),
    )

    summary = review_batch.create_review_bundles(
        corpus,
        min_free_bytes=0,
        max_bundle_bytes=1,
    )

    assert summary["blocked"] is True
    assert summary["projects"][0]["status"] == "bundle_budget_blocked"


def test_create_review_bundles_scales_preflight_to_actual_pending_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small frontier should fit when its actual request bound fits."""

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _project(corpus, "pending")
    monkeypatch.setattr(review_batch, "pending_packet_count", lambda _path: 1)
    monkeypatch.setattr(
        review_batch.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(total=100 * 1024**2, used=1, free=99 * 1024**2),
    )

    def fake_create(project_dir: Path, *, output_dir: Path | None, max_packets: int):
        assert max_packets == 8
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "bundle.json").write_text("{}", encoding="utf-8")
        return {"bundle_dir": str(output_dir), "bundle_id": "SBONE", "request_count": 1}

    monkeypatch.setattr(review_batch, "create_review_bundle", fake_create)
    one_packet_bound = review_batch.estimate_review_bundle_bytes(1)
    summary = review_batch.create_review_bundles(
        corpus,
        output_root=tmp_path / "handoff",
        max_packets_per_project=8,
        max_bundle_bytes=one_packet_bound,
        min_free_bytes=0,
    )

    record = summary["projects"][0]
    assert record["status"] == "created"
    assert record["estimated_packet_count"] == 1
    assert record["estimated_bundle_bytes"] == one_packet_bound


def test_create_review_bundles_measures_large_bound_before_accepting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conservative per-request ceiling must not block a measured small bundle."""

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _project(corpus, "pending")
    monkeypatch.setattr(review_batch, "pending_packet_count", lambda _path: 32)
    monkeypatch.setattr(
        review_batch.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(
            total=2 * 1024**3, used=1, free=2 * 1024**3 - 1
        ),
    )

    def fake_create(project_dir: Path, *, output_dir: Path | None, max_packets: int):
        assert max_packets == 32
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "bundle.json").write_text("{}", encoding="utf-8")
        return {"bundle_dir": str(output_dir), "bundle_id": "SBMEASURED", "request_count": 32}

    monkeypatch.setattr(review_batch, "create_review_bundle", fake_create)
    summary = review_batch.create_review_bundles(
        corpus,
        output_root=tmp_path / "handoff",
        max_packets_per_project=32,
        max_bundle_bytes=1 * 1024**2,
        min_free_bytes=0,
    )

    record = summary["projects"][0]
    assert record["status"] == "created"
    assert record["actual_preflight"] is True
    assert record["bundle_bytes"] == 2
    assert Path(record["bundle_dir"]).is_dir()
    assert not list((tmp_path / "handoff").glob(".*.preflight-*"))


def test_measured_bundle_over_budget_is_blocked_and_staging_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _project(corpus, "pending")
    monkeypatch.setattr(review_batch, "pending_packet_count", lambda _path: 32)
    monkeypatch.setattr(
        review_batch.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(
            total=2 * 1024**3, used=1, free=2 * 1024**3 - 1
        ),
    )

    def fake_create(project_dir: Path, *, output_dir: Path | None, max_packets: int):
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "oversize.bin").write_bytes(b"x" * (2 * 1024**2))
        return {"bundle_dir": str(output_dir), "bundle_id": "SBOVERSIZE", "request_count": 32}

    monkeypatch.setattr(review_batch, "create_review_bundle", fake_create)
    handoff = tmp_path / "handoff"
    summary = review_batch.create_review_bundles(
        corpus,
        output_root=handoff,
        max_packets_per_project=32,
        max_bundle_bytes=1 * 1024**2,
        min_free_bytes=0,
    )

    record = summary["projects"][0]
    assert record["status"] == "bundle_budget_blocked"
    assert record["measured_bundle_bytes"] == 2 * 1024**2
    assert not list(handoff.glob(".*.preflight-*"))
    assert not list(handoff.glob("*"))


def test_create_review_bundles_uses_stable_project_output_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _project(corpus, "same-name")
    monkeypatch.setattr(review_batch, "pending_packet_count", lambda _path: 1)
    monkeypatch.setattr(
        review_batch.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(
            total=100 * 1024**2, used=1, free=99 * 1024**2
        ),
    )

    def fake_create(project_dir: Path, *, output_dir: Path | None, max_packets: int):
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "bundle.json").write_text("{}", encoding="utf-8")
        return {"bundle_dir": str(output_dir), "bundle_id": "SBTEST", "request_count": 1}

    monkeypatch.setattr(review_batch, "create_review_bundle", fake_create)
    summary = review_batch.create_review_bundles(
        corpus,
        output_root=tmp_path / "handoff",
        min_free_bytes=1,
    )

    record = summary["projects"][0]
    assert record["status"] == "created"
    assert record["copied_media_bytes"] == 0
    assert Path(record["output_dir"]).name.startswith("same-name-")
    assert record["bundle_bytes"] == 2


def test_create_review_bundles_rejects_output_root_inside_project(tmp_path: Path) -> None:
    project = _project(tmp_path, "project")
    with pytest.raises(review_batch.InputError, match="inside a canonical project"):
        review_batch.create_review_bundles(
            project,
            output_root=project / "handoff",
            min_free_bytes=0,
        )
    assert not (project / "handoff").exists()


def test_create_review_bundles_counts_stale_default_bundle_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    project = _project(corpus, "pending")
    stale = project / ".state" / "vision" / "subagent-review"
    stale.mkdir(parents=True)
    max_bundle_bytes = review_batch.estimate_review_bundle_bytes(8)
    (stale / "oversize.bin").write_bytes(b"x" * (max_bundle_bytes + 1))
    monkeypatch.setattr(review_batch, "pending_packet_count", lambda _path: 1)
    monkeypatch.setattr(
        review_batch.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(total=100, used=1, free=99),
    )
    called = False

    def fail_create(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("an over-budget existing tree must not be written")

    monkeypatch.setattr(review_batch, "create_review_bundle", fail_create)
    summary = review_batch.create_review_bundles(
        corpus,
        min_free_bytes=0,
        max_bundle_bytes=max_bundle_bytes,
    )

    assert called is False
    assert summary["blocked"] is True
    assert summary["projects"][0]["status"] == "bundle_budget_blocked"
    assert summary["projects"][0]["existing_bundle_bytes"] == max_bundle_bytes + 1


def test_create_review_bundles_explicitly_selects_legacy_provider_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    project = _project(corpus, "legacy")
    canonical_path = project / ".state" / "canonical-project.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["visual_events"] = [
        {"event_id": "V000001", "annotation_provider": "llama.cpp-local"}
    ]
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    monkeypatch.setattr(review_batch, "pending_packet_count", lambda _path: 0)
    monkeypatch.setattr(
        review_batch.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(
            total=100 * 1024**2, used=1, free=99 * 1024**2
        ),
    )
    captured: dict[str, object] = {}

    def fake_create(project_dir: Path, **kwargs: object):
        captured["project_dir"] = project_dir
        captured.update(kwargs)
        output = kwargs["output_dir"]
        assert isinstance(output, Path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "bundle.json").write_text("{}", encoding="utf-8")
        return {"bundle_dir": str(output), "bundle_id": "SBLEGACY", "request_count": 1}

    monkeypatch.setattr(review_batch, "create_review_bundle", fake_create)
    summary = review_batch.create_review_bundles(
        corpus,
        output_root=tmp_path / "handoff",
        min_free_bytes=1,
        include_annotation_providers=["llama.cpp-local"],
    )

    record = summary["projects"][0]
    assert record["status"] == "created"
    assert record["legacy_provider_event_count"] == 1
    assert captured["include_annotation_providers"] == ("llama.cpp-local",)
    assert summary["policy"]["include_annotation_providers"] == ["llama.cpp-local"]
