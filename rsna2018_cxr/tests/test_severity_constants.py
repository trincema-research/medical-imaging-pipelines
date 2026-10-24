from rsna2018_cxr.severity.cli import COMMANDS, main
from rsna2018_cxr.severity.constants import (
    NORMAL_CLASS,
    SEVERITY_POSITIVE_CLASSES,
    SPECIALIST_CHECKPOINTS,
)


def test_three_specialist_slots():
    assert len(SPECIALIST_CHECKPOINTS) == 3
    assert {row["positive_class"] for row in SPECIALIST_CHECKPOINTS} == {"severity_1_low_mild"}


def test_severity_shortcuts_cover_s1_s2_s3():
    assert NORMAL_CLASS == "normal"
    assert set(SEVERITY_POSITIVE_CLASSES) == {"severity_1", "severity_2", "severity_3"}
    assert SEVERITY_POSITIVE_CLASSES["severity_1"] == "severity_1_low_mild"


def test_cli_help(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "nas" in out
    assert "retrain" in out
    assert set(COMMANDS) >= {"train", "nas", "retrain", "download-weights"}
