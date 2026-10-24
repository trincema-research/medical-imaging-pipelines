import pytest

from rsna2025_brain.constants import SERIES_DIRNAME
from rsna2025_brain.data.validate import main, parse_args, validate_raw
from rsna2025_brain.tests.helpers import UID_A, write_extracted_dump, write_label_csvs, write_series


def test_validate_cli_default_is_data_raw():
    args = parse_args([])
    assert args.data_root.name == "raw"


def test_validate_synthetic_dump_ok(tmp_path):
    write_extracted_dump(tmp_path)
    report = validate_raw(tmp_path)
    assert report.ok
    assert report.n_series == 2
    assert report.n_dicoms == 2
    assert report.n_series_dirs_ok == 2
    assert report.n_series_dirs_missing == 0


def test_validate_missing_root_fails(tmp_path):
    report = validate_raw(tmp_path / "absent")
    assert report.ok is False
    assert any("does not exist" in e for e in report.errors)


def test_validate_incomplete_layout_fails(tmp_path):
    report = validate_raw(tmp_path)
    assert report.ok is False
    assert any("missing file" in e for e in report.errors)


def test_validate_missing_series_is_warning(tmp_path):
    write_label_csvs(tmp_path)
    write_series(tmp_path, uids=(UID_A,))
    report = validate_raw(tmp_path)
    assert report.ok
    assert report.n_series_dirs_missing == 1
    assert any("missing" in w for w in report.warnings)


def test_validate_strict_fails_missing_series(tmp_path):
    write_label_csvs(tmp_path)
    write_series(tmp_path, uids=(UID_A,))
    report = validate_raw(tmp_path, strict=True)
    assert report.ok is False


def test_validate_cli_on_dump_exits_ok(tmp_path, capsys):
    write_extracted_dump(tmp_path)
    main(["--data-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "OK" in out
    assert "series:" in out


def test_validate_cli_rejects_empty(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--data-root", str(tmp_path)])
    assert exc.value.code == 1
    assert SERIES_DIRNAME
