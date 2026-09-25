from __future__ import annotations

from pathlib import Path

import pytest

from rsna2024_lumbar.tests.helpers import write_mini_dataset

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def mini_raw(tmp_path_factory) -> Path:
    """Fresh synthetic dump per test session (also written to tests/fixtures/)."""
    session = tmp_path_factory.mktemp("raw")
    write_mini_dataset(session)
    write_mini_dataset(FIXTURE_ROOT)
    return session
