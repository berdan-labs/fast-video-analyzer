from pathlib import Path
from types import SimpleNamespace

import video_script_reconstructor.batch as batch_module
from video_script_reconstructor.batch import (
    discover_sidecars,
    discover_videos,
    estimate_project_bytes,
    run_batch,
)


def test_discover_videos_is_recursive_and_deterministic(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "second.MP4").write_bytes(b"video")
    (tmp_path / "a" / "first.mkv").write_bytes(b"video")
    (tmp_path / "notes.txt").write_text("not media", encoding="utf-8")

    found = discover_videos(tmp_path)

    assert [path.name for path in found] == ["first.mkv", "second.MP4"]


def test_discover_videos_uses_scandir_and_does_not_follow_symlinks(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "sources"
    nested = source_root / "nested"
    nested.mkdir(parents=True)
    (nested / "lesson.mp4").write_bytes(b"video")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")

    def fail_rglob(*_args, **_kwargs):
        raise AssertionError("video discovery must use scandir")

    monkeypatch.setattr(Path, "rglob", fail_rglob)
    try:
        (source_root / "outside-link.mp4").symlink_to(outside)
        (source_root / "linked-dir").symlink_to(tmp_path / "linked", target_is_directory=True)
    except (OSError, NotImplementedError):
        pass

    found = discover_videos(source_root)

    assert found == ((nested / "lesson.mp4").resolve(),)


def test_historical_rates_uses_scandir_for_nested_history(tmp_path: Path, monkeypatch) -> None:
    history = tmp_path / "history"
    project = history / "nested" / "lesson"
    state = project / ".state"
    state.mkdir(parents=True)
    (state / "canonical-project.json").write_text(
        '{"media":{"duration_ms":60000}}', encoding="utf-8"
    )
    (state / "run-manifest.json").write_text(
        '{"performance":{"resource_usage":{"output":{"bytes":1000000}}}}',
        encoding="utf-8",
    )

    def fail_rglob(*_args, **_kwargs):
        raise AssertionError("history discovery must use scandir")

    monkeypatch.setattr(Path, "rglob", fail_rglob)

    rates = batch_module._historical_rates((history,))

    assert rates["short"] == [1000000 / 60]


def test_discover_sidecars_ignores_empty_and_unrelated_files(tmp_path: Path) -> None:
    source = tmp_path / "lesson.mp4"
    source.write_bytes(b"video")
    (tmp_path / "lesson.srt").write_text("1\n", encoding="utf-8")
    (tmp_path / "lesson.vtt").write_text("", encoding="utf-8")
    (tmp_path / "lesson.ass").write_text("[Script Info]", encoding="utf-8")
    (tmp_path / "lesson.txt").write_text("course notes", encoding="utf-8")

    assert [path.suffix for path in discover_sidecars(source)] == [".srt", ".ass"]


def test_tree_bytes_recurses_with_scandir_and_rejects_symlinks(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "output"
    nested = output / "nested"
    nested.mkdir(parents=True)
    (output / "root.bin").write_bytes(b"root")
    (nested / "child.bin").write_bytes(b"child")

    def fail_walk(*_args, **_kwargs):
        raise AssertionError("batch tree inventory must recurse with scandir")

    monkeypatch.setattr(batch_module.os, "walk", fail_walk)
    try:
        (output / "outside-link").symlink_to(tmp_path / "outside.bin")
    except (OSError, NotImplementedError):
        pass

    assert batch_module._tree_bytes(output) == len(b"root") + len(b"child")


def test_estimate_project_bytes_is_duration_aware_and_bounded() -> None:
    short = estimate_project_bytes(60_000, max_project_bytes=100_000_000)
    long = estimate_project_bytes(3_600_000, max_project_bytes=100_000_000)

    assert short >= 64 * 1024**2
    assert long == 100_000_000


def test_estimate_project_bytes_accepts_a_precomputed_history_snapshot(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        batch_module,
        "_historical_rates",
        lambda _roots: (_ for _ in ()).throw(AssertionError("history rescanned")),
    )

    forecast = estimate_project_bytes(
        60_000,
        historical_rates={"short": [10_000_000.0], "medium": [], "long": []},
        max_project_bytes=100_000_000,
    )

    assert forecast == 100_000_000


def test_batch_dry_run_scans_history_once_for_multiple_sources(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    (source_root / "first.mp4").write_bytes(b"video")
    (source_root / "second.mp4").write_bytes(b"video")
    calls = 0

    def fake_history(_roots):
        nonlocal calls
        calls += 1
        return {"short": [], "medium": [], "long": []}

    monkeypatch.setattr(batch_module, "_historical_rates", fake_history)
    monkeypatch.setattr(
        batch_module,
        "probe_media",
        lambda _path: SimpleNamespace(duration_ms=120_000, size_bytes=11),
    )

    summary = run_batch(
        source_root,
        output_root=tmp_path / "out",
        min_free_bytes=0,
        dry_run=True,
    )

    assert summary["blocked"] is False
    assert len(summary["planned"]) == 2
    assert calls == 1


def test_batch_dry_run_writes_a_plan_without_running_pipeline(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    (source_root / "lesson.mp4").write_bytes(b"placeholder")
    output_root = tmp_path / "out"
    monkeypatch.setattr(
        batch_module,
        "probe_media",
        lambda _path: SimpleNamespace(duration_ms=120_000, size_bytes=11),
    )

    summary = run_batch(
        source_root,
        output_root=output_root,
        min_free_bytes=0,
        vision_mode="none",
        dry_run=True,
    )

    assert summary["blocked"] is False
    assert summary["executed"] == []
    assert summary["planned"][0]["source_bytes"] == 11
    assert (output_root / ".challenge-batch.json").is_file()


def test_batch_without_output_keeps_projects_beside_sources(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    (source_root / "lesson.mp4").write_bytes(b"placeholder")
    monkeypatch.delenv("VSR_OUTPUT_ROOT", raising=False)
    monkeypatch.setattr(
        batch_module,
        "probe_media",
        lambda _path: SimpleNamespace(duration_ms=120_000, size_bytes=11),
    )

    summary = run_batch(source_root, min_free_bytes=0, vision_mode="none", dry_run=True)

    expected = source_root / "lesson (Analyzer Outputs)"
    assert summary["planned"][0]["project_dir"] == str(expected)
    assert (source_root / "(Analyzer Batch Outputs)" / ".challenge-batch.json").is_file()


def test_batch_forwards_explicit_language_hint_to_each_run(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "lesson.mp4"
    source.write_bytes(b"video")
    output_root = tmp_path / "out"
    monkeypatch.setattr(
        batch_module,
        "probe_media",
        lambda _path: SimpleNamespace(duration_ms=120_000, size_bytes=11),
    )

    class _Validation:
        valid = True
        errors: list[str] = []

    class _Result:
        project_dir = output_root / "lesson"
        markdown_path = project_dir / "lesson.reconstruction.md"
        status = "automatically_checked"
        exit_code = 0
        validation = _Validation()

    _Result.project_dir.mkdir(parents=True)
    (_Result.project_dir / "lesson.reconstruction.md").write_text("# lesson", encoding="utf-8")
    seen: list[str | None] = []

    def fake_run_pipeline(*_args, **kwargs):
        seen.append(kwargs.get("language"))
        return _Result()

    monkeypatch.setattr(batch_module, "run_pipeline", fake_run_pipeline)

    summary = run_batch(
        source_root,
        output_root=output_root,
        language="fil",
        min_free_bytes=0,
    )

    assert summary["blocked"] is False
    assert seen == ["fil"]
    assert summary["policy"]["language"] == "fil"


def test_batch_compare_sidecars_forwards_adjacent_subtitles(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "lesson.mp4"
    source.write_bytes(b"video")
    sidecar = source.with_suffix(".srt")
    sidecar.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello.\n", encoding="utf-8")
    output_root = tmp_path / "out"
    monkeypatch.setattr(
        batch_module,
        "probe_media",
        lambda _path: SimpleNamespace(duration_ms=120_000, size_bytes=11),
    )

    class _Validation:
        valid = True
        errors: list[str] = []

    class _Result:
        project_dir = output_root / "lesson"
        markdown_path = project_dir / "lesson.reconstruction.md"
        status = "automatically_checked"
        exit_code = 0
        validation = _Validation()

    _Result.project_dir.mkdir(parents=True)
    (_Result.project_dir / "lesson.reconstruction.md").write_text("# lesson", encoding="utf-8")
    seen: list[dict[str, object]] = []

    def fake_run_pipeline(*_args, **kwargs):
        seen.append(kwargs)
        return _Result()

    monkeypatch.setattr(batch_module, "run_pipeline", fake_run_pipeline)

    summary = run_batch(
        source_root,
        output_root=output_root,
        compare_sidecars=True,
        min_free_bytes=0,
    )

    assert summary["blocked"] is False
    assert seen[0]["subtitle_mode"] == "compare-all"
    assert seen[0]["subtitles"] == (sidecar.resolve(),)
    assert summary["planned"][0]["sidecars"] == [str(sidecar.resolve())]
