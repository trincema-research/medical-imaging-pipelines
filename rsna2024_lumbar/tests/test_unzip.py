import subprocess
import zipfile
from pathlib import Path

import pytest

from rsna2024_lumbar.data.unzip import main
from rsna2024_lumbar.preprocessing.constants import LABEL_CSVS


def _write_dump_zip(archive: Path) -> Path:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as zf:
        for name in LABEL_CSVS:
            zf.writestr(name, "study_id\n")
        zf.writestr("train_images/1/2/1.dcm", b"dicom")
    return archive


def test_unzip_extracts_and_deletes_zip(tmp_path, capsys):
    _write_dump_zip(tmp_path / "comp.zip")
    main(["--dest", str(tmp_path)])
    out = capsys.readouterr().out
    assert (tmp_path / "train.csv").is_file()
    assert (tmp_path / "train_images").is_dir()
    assert not list(tmp_path.glob("*.zip"))
    assert "Unzip OK" in out
    assert "data.validate" in out


def test_unzip_keep_zip(tmp_path):
    _write_dump_zip(tmp_path / "comp.zip")
    main(["--dest", str(tmp_path), "--keep-zip"])
    assert (tmp_path / "comp.zip").is_file()
    assert (tmp_path / "train.csv").is_file()


def test_unzip_already_extracted(tmp_path, capsys):
    (tmp_path / "train.csv").write_text("study_id\n", encoding="utf-8")
    (tmp_path / "train_label_coordinates.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "train_series_descriptions.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "train_images").mkdir()
    main(["--dest", str(tmp_path)])
    assert "Already extracted" in capsys.readouterr().out


def test_unzip_fails_without_zip_or_dump(tmp_path):
    with pytest.raises(SystemExit, match="Unzip failed"):
        main(["--dest", str(tmp_path)])


def test_unzip_fails_on_partial(tmp_path):
    (tmp_path / "comp.zip.kaggle-partial").write_bytes(b"x")
    with pytest.raises(SystemExit, match="incomplete"):
        main(["--dest", str(tmp_path)])


def test_unzip_fails_on_corrupt_zip(tmp_path):
    (tmp_path / "comp.zip").write_bytes(b"not-a-zip")
    with pytest.raises(SystemExit, match="Unzip failed"):
        main(["--dest", str(tmp_path)])


def test_download_reports_cli_failure(tmp_path, monkeypatch):
    from common.utils import kaggle as kaggle_mod
    from rsna2024_lumbar.data.download import main as download_main

    monkeypatch.setattr(kaggle_mod, "ensure_kaggle_token", lambda: None)
    monkeypatch.setattr(kaggle_mod, "require_kaggle_cli", lambda: "kaggle")
    monkeypatch.setattr(
        kaggle_mod.subprocess,
        "run",
        lambda cmd, check=False: subprocess.CompletedProcess(cmd, 1),
    )
    with pytest.raises(SystemExit, match="download failed"):
        download_main(["--dest", str(tmp_path), "--force"])


def test_download_reports_partial_after_cli(tmp_path, monkeypatch):
    from common.utils import kaggle as kaggle_mod
    from rsna2024_lumbar.data.download import main as download_main

    def fake_run(cmd, check=False):
        (tmp_path / "comp.zip.kaggle-partial").write_bytes(b"x")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(kaggle_mod, "ensure_kaggle_token", lambda: None)
    monkeypatch.setattr(kaggle_mod, "require_kaggle_cli", lambda: "kaggle")
    monkeypatch.setattr(kaggle_mod.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="incomplete"):
        download_main(["--dest", str(tmp_path), "--force"])


def test_download_unzips_and_reports_ok(tmp_path, monkeypatch, capsys):
    from common.utils import kaggle as kaggle_mod
    from rsna2024_lumbar.data.download import main as download_main

    def fake_run(cmd, check=False):
        _write_dump_zip(tmp_path / "comp.zip")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(kaggle_mod, "ensure_kaggle_token", lambda: None)
    monkeypatch.setattr(kaggle_mod, "require_kaggle_cli", lambda: "kaggle")
    monkeypatch.setattr(kaggle_mod.subprocess, "run", fake_run)
    download_main(["--dest", str(tmp_path), "--force"])
    out = capsys.readouterr().out
    assert (tmp_path / "train.csv").is_file()
    assert not list(tmp_path.glob("*.zip"))
    assert "Download OK" in out
    assert "Download + unzip OK" in out
    assert "data.validate" in out
