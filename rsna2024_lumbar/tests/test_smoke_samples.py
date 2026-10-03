import pytest

from pathlib import Path

SMOKE_STUDIES = (3318343342, 3065863143)
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
CENTERED_MANIFEST = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "centered" / "manifest.csv"
)


def test_committed_smoke_studies_have_dicoms():
    for study_id in SMOKE_STUDIES:
        dcms = list((RAW / "train_images" / str(study_id)).rglob("*.dcm"))
        assert len(dcms) >= 15, f"study {study_id} missing smoke DICOMs"


@pytest.mark.local
def test_centered_crop_cache_has_manifest():
    """Full PNG cache is local-only (~240MB, not in git / CI)."""
    assert CENTERED_MANIFEST.is_file()
