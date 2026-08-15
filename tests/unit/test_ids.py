from __future__ import annotations

import pytest

from video_script_reconstructor.ids import (
    IdentifierError,
    StableIdAllocator,
    crop_id,
    deterministic_id,
    evidence_filename,
    media_id,
    parse_id,
    sequential_id,
    timestamp_slug,
    validate_id,
)


def test_contract_ids_and_filenames_are_stable() -> None:
    assert sequential_id("transcript", 1) == "T000001"
    assert sequential_id("chapter", 12) == "C012"
    assert crop_id("F000042", 1) == "F000042-C01"
    assert timestamp_slug(3_807_415) == "01h03m27s415"
    assert evidence_filename("F000042", 3_807_415) == "F000042__01h03m27s415__full.png"
    assert evidence_filename("F000042-C01", 3_807_415) == "F000042-C01__01h03m27s415__detail.png"


def test_content_ids_and_allocator_survive_reruns() -> None:
    digest = "a" * 64
    assert media_id(digest) == "M" + "A" * 16
    assert deterministic_id("image_claim", "frame", "text") == deterministic_id(
        "image_claim", "frame", "text"
    )
    allocator = StableIdAllocator("frame", {"first": "F000001"})
    assert allocator.assign_all(["first", "second"]) == ["F000001", "F000002"]
    resumed = StableIdAllocator("frame", dict(allocator.assignments))
    assert resumed.assign_all(["first", "second", "third"]) == [
        "F000001",
        "F000002",
        "F000003",
    ]


def test_validation_rejects_wrong_width_zero_and_bad_crop() -> None:
    assert validate_id("VA000001", "visual_analysis")
    assert parse_id("MR000003") == ("metadata_revision", 3)
    assert not validate_id("T1")
    assert not validate_id("T000000")
    with pytest.raises(IdentifierError):
        crop_id("F000001", 100)
