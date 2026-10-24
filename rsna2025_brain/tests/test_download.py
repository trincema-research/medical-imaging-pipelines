from pathlib import Path

import pytest

from rsna2025_brain.data.download import (
    DEFAULT_RAW,
    KAGGLE_SLUG,
    main,
    parse_args,
    raw_looks_complete,
)
from rsna2025_brain.tests.helpers import UID_A, UID_B, write_extracted_dump, write_label_csvs, write_series


def test_default_dest_is_data_raw():
    args = parse_args([])
    assert args.dest == DEFAULT_RAW
    assert args.dest.name == "raw"
    assert args.dest.parent.name == "data"
    assert (args.dest.parent / "download.py").is_file()


def test_slug_is_the_2025_brain_competition():
    assert KAGGLE_SLUG == "rsna-intracranial-aneurysm-detection"


def test_skips_when_raw_already_complete(tmp_path, capsys):
    write_extracted_dump(tmp_path)
    main(["--dest", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Already present" in out
    assert raw_looks_complete(tmp_path)


def test_incomplete_raw_is_not_skipped(tmp_path):
    write_label_csvs(tmp_path, both_series=True)
    write_series(tmp_path, uids=(UID_A,))
    assert raw_looks_complete(tmp_path) is False
    assert UID_B


def test_rejects_download_without_credentials(tmp_path, monkeypatch):
    from common.utils import kaggle as kaggle_mod

    monkeypatch.setattr(kaggle_mod, "ensure_kaggle_token", lambda: None)
    monkeypatch.setattr(kaggle_mod, "_download_via_api", lambda *a, **k: False)
    monkeypatch.setattr(kaggle_mod.shutil, "which", lambda _name: None)
    with pytest.raises(SystemExit, match="kaggle CLI"):
        main(["--dest", str(tmp_path), "--force"])
