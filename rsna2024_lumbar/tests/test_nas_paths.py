"""NAS path helpers (data roots and bundle layout)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rsna2024_lumbar.nas.export_compact import parse_args as export_parse_args
from rsna2024_lumbar.nas.paths import (
    default_crops_root,
    default_data_root,
    repo_root,
)


def test_default_data_root_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("RSNA2024_DATA_ROOT", str(tmp_path))
    assert default_data_root() == tmp_path.resolve()


def test_default_crops_root_points_under_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("RSNA2024_CROPS_ROOT", str(tmp_path / "crops"))
    assert default_crops_root("centered") == (tmp_path / "crops").resolve()


def test_export_compact_default_output_under_data_module():
    args = export_parse_args([])
    assert args.output_root == Path("rsna2024_lumbar/data/nas_compact")


def test_repo_root_contains_pyproject():
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
