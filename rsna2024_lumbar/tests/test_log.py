from common.utils.log import (
    data,
    done,
    fmt_bytes,
    infer_traces_dir,
    next_step,
    ok,
    start,
    step,
    trace_paths,
)
from rsna2024_lumbar import TRACES_DIR, YEAR_ROOT


def test_fmt_bytes():
    assert fmt_bytes(512) == "512 B"
    assert fmt_bytes(2048).endswith("KB")
    assert "MB" in fmt_bytes(2 * 1024 * 1024)


def test_trace_kinds_on_stdout(capsys):
    start("rsna2024.data.download", dest="raw")
    step("inspect dest")
    data(zips="(none)")
    ok("Already present: /tmp/raw")
    next_step("python -m rsna2024_lumbar.data.validate")
    done("download (skipped)")
    out = capsys.readouterr().out
    for kind in ("START", "STEP", "DATA", "OK", "NEXT", "DONE"):
        assert kind in out
    assert "rsna2024.data.download" in out
    assert "validate" in out


def test_download_skip_emits_trace(tmp_path, capsys):
    from rsna2024_lumbar.data.download import main

    (tmp_path / "train.csv").write_text("study_id\n1\n", encoding="utf-8")
    (tmp_path / "train_images" / "1").mkdir(parents=True)
    main(["--dest", str(tmp_path)])
    out = capsys.readouterr().out
    assert "START" in out
    assert "Already present" in out
    assert "NEXT" in out
    assert "DONE" in out


def test_traces_dir_is_under_rsna2024_lumbar():
    assert TRACES_DIR == YEAR_ROOT / "traces"
    assert infer_traces_dir("rsna2024.data.download") == TRACES_DIR
    paths = trace_paths(TRACES_DIR, "rsna2024.data.download")
    assert paths[0].name == "download.log"
    assert paths[1].name == "pipeline.log"


def test_traces_write_component_and_pipeline(tmp_path, capsys):
    start("rsna2024.data.download", traces_dir=tmp_path, force=True)
    step("inspect dest")
    done("download")
    capsys.readouterr()
    component = (tmp_path / "download.log").read_text(encoding="utf-8")
    pipeline = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
    assert "--------" in component and "download" in component
    assert "START" in component and "DONE" in component
    assert component == pipeline


def test_validate_cli_emits_trace(mini_raw, capsys):
    from rsna2024_lumbar.data.validate import main

    main(["--data-root", str(mini_raw)])
    out = capsys.readouterr().out
    assert "START" in out
    assert "studies:" in out
    assert "OK" in out
    assert "preprocessing" in out
    assert "DONE" in out
