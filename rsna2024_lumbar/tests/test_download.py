from pathlib import Path

import pytest

from rsna2024_lumbar.data.download import (
    DEFAULT_RAW,
    KAGGLE_SLUG,
    main,
    parse_args,
    raw_looks_complete,
)


def test_default_dest_is_data_raw_not_inside_gitignore_hole():
    args = parse_args([])
    assert args.dest == DEFAULT_RAW
    assert args.dest.name == "raw"
    assert args.dest.parent.name == "data"
    assert (args.dest.parent / "download.py").is_file()


def test_slug_is_the_2024_lumbar_competition():
    assert KAGGLE_SLUG == "rsna-2024-lumbar-spine-degenerative-classification"


def test_skips_when_raw_already_complete(tmp_path, capsys):
    (tmp_path / "train.csv").write_text("study_id\n", encoding="utf-8")
    (tmp_path / "train_images").mkdir()
    main(["--dest", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Already present" in out
    assert raw_looks_complete(tmp_path)


def test_incomplete_raw_is_not_skipped(tmp_path):
    (tmp_path / "train.csv").write_text("study_id\n", encoding="utf-8")
    assert raw_looks_complete(tmp_path) is False


def test_rejects_download_without_credentials(tmp_path, monkeypatch):
    from common.utils import kaggle as kaggle_mod

    monkeypatch.setattr(kaggle_mod, "_download_via_api", lambda *a, **k: False)
    monkeypatch.setattr(kaggle_mod.shutil, "which", lambda _name: None)
    with pytest.raises(SystemExit, match="kaggle CLI"):
        main(["--dest", str(tmp_path), "--force"])
