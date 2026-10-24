from common.utils.log import infer_traces_dir, start, step, done, trace_paths
from rsna2025_brain import TRACES_DIR, YEAR_ROOT


def test_traces_dir_is_under_rsna2025_brain():
    assert TRACES_DIR == YEAR_ROOT / "traces"
    assert infer_traces_dir("rsna2025.data.download") == TRACES_DIR
    paths = trace_paths(TRACES_DIR, "rsna2025.data.download")
    assert paths[0].name == "download.log"
    assert paths[1].name == "pipeline.log"


def test_download_skip_emits_trace(tmp_path, capsys):
    from rsna2025_brain.data.download import main
    from rsna2025_brain.tests.helpers import write_extracted_dump

    write_extracted_dump(tmp_path)
    main(["--dest", str(tmp_path)])
    out = capsys.readouterr().out
    assert "START" in out
    assert "Already present" in out
    assert "NEXT" in out
    assert "DONE" in out


def test_traces_write_component_and_pipeline(tmp_path, capsys):
    start("rsna2025.data.download", traces_dir=tmp_path, force=True)
    step("inspect dest")
    done("download")
    capsys.readouterr()
    component = (tmp_path / "download.log").read_text(encoding="utf-8")
    pipeline = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
    assert "--------" in component and "download" in component
    assert "START" in component and "DONE" in component
    assert component == pipeline
