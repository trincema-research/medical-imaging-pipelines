from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def mini_raw() -> Path:
    """Committed Kaggle-shaped mini-dump. CI and local tests use this folder."""
    csv_path = FIXTURE_ROOT / "train.csv"
    images = FIXTURE_ROOT / "train_images"
    if not csv_path.is_file() or not images.is_dir():
        raise RuntimeError(
            f"Missing committed fixtures under {FIXTURE_ROOT}. "
            "Regenerate with: python -m rsna2024_lumbar.tests.helpers"
        )
    return FIXTURE_ROOT
