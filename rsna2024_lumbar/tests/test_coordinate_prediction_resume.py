"""Crash-resume for landmark NAS / train (no GPU, no DICOM dump)."""

from __future__ import annotations

from pathlib import Path

from rsna2024_lumbar.coordinate_prediction.launch import (
    main as launch_main,
    parse_args as parse_launch_args,
    spawn_shard,
)
from rsna2024_lumbar.coordinate_prediction.resume import (
    STATE_COMPLETE,
    STATE_INCOMPLETE,
    STATE_PENDING,
    append_training_metrics_row,
    atomic_write_text,
    classify_trial,
    collect_trial_statuses,
    flatten_epoch_metrics,
    format_id_ranges,
    format_status_report,
    history_path,
    last_checkpoint_path,
    last_logged_epoch,
    next_epoch,
    resume_hparams,
    resume_hparams_match,
    training_metrics_path,
    trial_is_complete,
    trial_output_dir,
    wants_resume,
    write_epoch_history_csvs,
)
from rsna2024_lumbar.coordinate_prediction.search import expand_trials
from rsna2024_lumbar.coordinate_prediction.worker import main as worker_main
from rsna2024_lumbar.nas.gpus import GpuShard


def _hparams(**overrides) -> dict:
    base = resume_hparams(
        condition="spinal_canal_stenosis",
        backbone="convnext_tiny",
        image_size=512,
        batch_size=8,
        lr=1e-4,
        weight_decay=0.01,
        seed=42,
        pretrained=True,
        all_slices=False,
    )
    base.update(overrides)
    return base


def test_resume_hparams_match_ignores_epochs():
    saved = _hparams()
    current = _hparams()
    assert resume_hparams_match(saved, current)
    assert not resume_hparams_match(saved, _hparams(lr=3e-4))
    assert not resume_hparams_match({}, current)
    assert not resume_hparams_match(None, current)


def test_next_epoch_continues_after_last_finished():
    assert next_epoch(0) == 1
    assert next_epoch(7) == 8
    assert next_epoch(-1) == 1


def test_trial_complete_only_when_history_exists(tmp_path: Path):
    condition, backbone = "spinal_canal_stenosis", "convnext_tiny"
    assert classify_trial(tmp_path, condition, backbone, epochs=20) == STATE_PENDING
    last_checkpoint_path(tmp_path, condition, backbone).write_bytes(b"partial")
    assert trial_is_complete(tmp_path, condition, backbone, epochs=20) is False
    assert classify_trial(tmp_path, condition, backbone, epochs=20) == STATE_INCOMPLETE
    atomic_write_text(history_path(tmp_path, condition, backbone), "{}")
    assert trial_is_complete(tmp_path, condition, backbone, epochs=20) is True
    assert classify_trial(tmp_path, condition, backbone, epochs=20) == STATE_COMPLETE


def test_metrics_csv_completes_when_planned_epochs_are_logged(tmp_path: Path):
    condition, backbone = "spinal_canal_stenosis", "convnext_tiny"
    csv_path = training_metrics_path(tmp_path)
    append_training_metrics_row(csv_path, {"epoch": 1, "val_l1": 0.4})
    append_training_metrics_row(csv_path, {"epoch": 2, "val_l1": 0.3})
    assert last_logged_epoch(csv_path) == 2
    assert trial_is_complete(tmp_path, condition, backbone, epochs=3) is False
    assert classify_trial(tmp_path, condition, backbone, epochs=3) == STATE_INCOMPLETE
    append_training_metrics_row(csv_path, {"epoch": 3, "val_l1": 0.2})
    assert trial_is_complete(tmp_path, condition, backbone, epochs=3) is True
    assert classify_trial(tmp_path, condition, backbone, epochs=3) == STATE_COMPLETE


def test_flatten_and_write_epoch_csvs(tmp_path: Path):
    metrics = {
        "n": 4,
        "loss": 0.2,
        "bce": 0.1,
        "n_visible": 3,
        "l1": 0.05,
        "r2": 0.9,
        "r2_x": 0.8,
        "r2_y": 0.7,
        "mae_px": 2.5,
        "mae_x_px": 1.0,
        "mae_y_px": 1.5,
        "pck5": 0.5,
        "pck10": 0.8,
        "levels": {},
    }
    row = flatten_epoch_metrics(1, {"train": metrics, "val": metrics, "test": metrics})
    assert row["epoch"] == 1
    assert row["val_l1"] == 0.05
    write_epoch_history_csvs(tmp_path, row)
    assert last_logged_epoch(training_metrics_path(tmp_path)) == 1
    assert (tmp_path / "training_history.csv").is_file()


def test_format_id_ranges_collapses_contiguous_ids():
    assert format_id_ranges([]) == "-"
    assert format_id_ranges([3]) == "3"
    assert format_id_ranges([1, 2, 3, 5, 7, 8]) == "1-3, 5, 7-8"


def test_status_report_lists_complete_and_incomplete(tmp_path: Path):
    trials = expand_trials("spinal_canal_stenosis", family="local")[:3]
    first = trial_output_dir(tmp_path, "local", trials[0].condition, trials[0].slug)
    first.mkdir(parents=True)
    (first / f"{trials[0].condition}_{trials[0].backbone}_history.json").write_text("{}", encoding="utf-8")
    second = trial_output_dir(tmp_path, "local", trials[1].condition, trials[1].slug)
    second.mkdir(parents=True)
    (second / "trial.json").write_text("{}", encoding="utf-8")
    rows = collect_trial_statuses(trials, output_base=tmp_path, family="local")
    assert [row.state for row in rows] == [STATE_COMPLETE, STATE_INCOMPLETE, STATE_PENDING]
    text = format_status_report(rows, family="local", output_base=tmp_path)
    assert "1/3 complete  1 incomplete  1 pending" in text
    assert "complete    1" in text
    assert f"incomplete  2  {trials[1].slug}" in text
    assert "pending     3" in text


def test_launch_status_uses_output_base(tmp_path: Path, capsys):
    trial = expand_trials("spinal_canal_stenosis", family="local")[0]
    out = trial_output_dir(tmp_path, "local", trial.condition, trial.slug)
    out.mkdir(parents=True)
    (out / f"{trial.condition}_{trial.backbone}_history.json").write_text("{}", encoding="utf-8")
    assert (
        launch_main(
            [
                "--family",
                "local",
                "--status",
                "--output-base",
                str(tmp_path),
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "family=local" in text
    assert "1/24 complete" in text
    assert "complete    1" in text
    assert "pending     2-24" in text


def test_atomic_write_replaces_without_leaving_tmp(tmp_path: Path):
    dest = tmp_path / "history.json"
    atomic_write_text(dest, '{"ok": true}')
    assert dest.read_text(encoding="utf-8") == '{"ok": true}'
    assert not dest.with_name(dest.name + ".tmp").exists()


def test_wants_resume_default_and_no_resume_flag():
    assert wants_resume(parse_launch_args([])) is True
    assert wants_resume(parse_launch_args(["--skip-completed"])) is True
    assert wants_resume(parse_launch_args(["--no-resume"])) is False


def test_spawn_shard_forwards_resume_by_default(monkeypatch, tmp_path: Path):
    captured: list[list[str]] = []

    class _Fake:
        def __init__(self, cmd, env=None):
            captured.append(list(cmd))

    monkeypatch.setattr("rsna2024_lumbar.coordinate_prediction.launch.subprocess.Popen", _Fake)
    args = parse_launch_args(["--output-base", str(tmp_path), "--data-root", str(tmp_path)])
    shard = GpuShard(gpu_id=0, start_trial=1, end_trial=2)
    plan = type("P", (), {"family": "local", "condition": "spinal_canal_stenosis"})()
    spawn_shard(plan, shard, args)
    assert "--resume" in captured[0]
    assert "--no-resume" not in captured[0]


def test_spawn_shard_forwards_no_resume(monkeypatch, tmp_path: Path):
    captured: list[list[str]] = []

    class _Fake:
        def __init__(self, cmd, env=None):
            captured.append(list(cmd))

    monkeypatch.setattr("rsna2024_lumbar.coordinate_prediction.launch.subprocess.Popen", _Fake)
    args = parse_launch_args(["--no-resume", "--output-base", str(tmp_path), "--data-root", str(tmp_path)])
    shard = GpuShard(gpu_id=0, start_trial=1, end_trial=2)
    plan = type("P", (), {"family": "local", "condition": "spinal_canal_stenosis"})()
    spawn_shard(plan, shard, args)
    assert "--no-resume" in captured[0]


def test_worker_skips_complete_metrics_csv(monkeypatch, tmp_path: Path, capsys):
    trial = expand_trials("spinal_canal_stenosis", family="local")[0]
    out = tmp_path / "local" / trial.condition / trial.slug
    out.mkdir(parents=True)
    for epoch in range(1, trial.epochs + 1):
        append_training_metrics_row(training_metrics_path(out), {"epoch": epoch, "val_l1": 0.1})
    called: list[str] = []

    def _boom(*_args, **_kwargs):
        called.append("train")
        raise AssertionError("complete csv trials must not retrain")

    monkeypatch.setattr("rsna2024_lumbar.coordinate_prediction.worker.train_condition", _boom)
    worker_main(
        [
            "--family",
            "local",
            "--condition",
            "spinal_canal_stenosis",
            "--start-trial",
            "1",
            "--end-trial",
            "1",
            "--output-base",
            str(tmp_path),
            "--data-root",
            str(tmp_path),
        ]
    )
    assert called == []
    assert "skip" in capsys.readouterr().out


def test_worker_skips_complete_trial(monkeypatch, tmp_path: Path, capsys):
    trial = expand_trials("spinal_canal_stenosis", family="local")[0]
    out = tmp_path / "local" / trial.condition / trial.slug
    out.mkdir(parents=True)
    (out / f"{trial.condition}_{trial.backbone}_history.json").write_text("{}", encoding="utf-8")
    called: list[str] = []

    def _boom(*_args, **_kwargs):
        called.append("train")
        raise AssertionError("complete trials must not retrain")

    monkeypatch.setattr("rsna2024_lumbar.coordinate_prediction.worker.train_condition", _boom)
    worker_main(
        [
            "--family",
            "local",
            "--condition",
            "spinal_canal_stenosis",
            "--start-trial",
            "1",
            "--end-trial",
            "1",
            "--output-base",
            str(tmp_path),
            "--data-root",
            str(tmp_path),
        ]
    )
    assert called == []
    assert "skip" in capsys.readouterr().out


def test_worker_resumes_incomplete_trial(monkeypatch, tmp_path: Path):
    trial = expand_trials("spinal_canal_stenosis", family="local")[0]
    seen: dict = {}

    def _fake(*_args, **kwargs):
        seen.update(kwargs)
        return {"splits": {}, "metrics_csv": None, "metrics_json": None}

    monkeypatch.setattr("rsna2024_lumbar.coordinate_prediction.worker.train_condition", _fake)
    worker_main(
        [
            "--family",
            "local",
            "--condition",
            "spinal_canal_stenosis",
            "--start-trial",
            "1",
            "--end-trial",
            "1",
            "--output-base",
            str(tmp_path),
            "--data-root",
            str(tmp_path),
        ]
    )
    assert seen["resume"] is True
    trial_dir = tmp_path / "local" / trial.condition / trial.slug
    assert (trial_dir / "trial.json").is_file()


def test_worker_no_resume_retrains_complete_trial(monkeypatch, tmp_path: Path):
    trial = expand_trials("spinal_canal_stenosis", family="local")[0]
    out = tmp_path / "local" / trial.condition / trial.slug
    out.mkdir(parents=True)
    (out / f"{trial.condition}_{trial.backbone}_history.json").write_text("{}", encoding="utf-8")
    seen: dict = {}

    def _fake(*_args, **kwargs):
        seen.update(kwargs)
        return {"splits": {}}

    monkeypatch.setattr("rsna2024_lumbar.coordinate_prediction.worker.train_condition", _fake)
    worker_main(
        [
            "--family",
            "local",
            "--condition",
            "spinal_canal_stenosis",
            "--start-trial",
            "1",
            "--end-trial",
            "1",
            "--output-base",
            str(tmp_path),
            "--data-root",
            str(tmp_path),
            "--no-resume",
        ]
    )
    assert seen["resume"] is False
