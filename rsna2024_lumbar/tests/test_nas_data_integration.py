"""Integration: legacy archive → compact export under data/ → summarize CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rsna2024_lumbar.nas.export_compact import main as export_main
from rsna2024_lumbar.nas.summarize import main as summarize_main
from rsna2024_lumbar.preprocessing.constants import CONDITIONS
from rsna2024_lumbar.tests.nas_fixtures import write_legacy_archive_trial


def _seed_convnext_2d_archive(archive: Path) -> None:
    """One winning trial per condition (best mode exports 5 folders)."""
    for i, condition in enumerate(CONDITIONS, start=1):
        write_legacy_archive_trial(
            archive,
            model_folder="Convnext",
            layout="2d",
            condition=condition,
            trial_dir_name=f"trial_{i}",
            trial_id=i,
            model_type="convnext",
            val_acc_at_best=0.5 + i * 0.01,
        )
        trial = archive / "Convnext" / "2d" / condition / f"trial_{i}"
        (trial / "training_history.csv").write_text(
            f"epoch,val_acc\n1,{0.5 + i * 0.01}\n",
            encoding="utf-8",
        )


def test_export_compact_to_data_layout_matches_summarize(tmp_path: Path):
    archive = tmp_path / "NAS results"
    data_out = tmp_path / "rsna2024_lumbar" / "data" / "nas_compact"
    _seed_convnext_2d_archive(archive)

    export_main(
        [
            "--archive-root",
            str(archive),
            "--output-root",
            str(data_out),
            "--models",
            "convnext",
            "--layouts",
            "2d",
            "--mode",
            "best",
        ]
    )

    manifest = json.loads((data_out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "best"
    assert manifest["output_root"].endswith("nas_compact")
    assert (
        data_out
        / "best"
        / "Convnext"
        / "2d"
        / "spinal_canal_stenosis"
        / "trial_0001"
        / "trial_config.json"
    ).is_file()

    summarize_dir = tmp_path / "summarize_out"
    summarize_main(
        [
            "--archive-root",
            str(archive),
            "--models",
            "convnext",
            "--layouts",
            "2d",
            "--csv-dir",
            str(summarize_dir),
        ]
    )

    export_csv = data_out / "best_configs" / "nas_best_convnext_2d.csv"
    summarize_csv = summarize_dir / "nas_best_convnext_2d.csv"
    assert export_csv.is_file() and summarize_csv.is_file()

    def trial_ids(path: Path) -> list[int]:
        with path.open(encoding="utf-8") as fh:
            return sorted(int(r["trial_id"]) for r in csv.DictReader(fh))

    assert trial_ids(export_csv) == trial_ids(summarize_csv) == list(range(1, 6))


def test_export_estimate_only_no_writes(tmp_path: Path):
    archive = tmp_path / "NAS results"
    _seed_convnext_2d_archive(archive)
    data_out = tmp_path / "compact"
    export_main(
        [
            "--archive-root",
            str(archive),
            "--output-root",
            str(data_out),
            "--models",
            "convnext",
            "--layouts",
            "2d",
            "--estimate-only",
        ]
    )
    assert not (data_out / "manifest.json").exists()


def test_export_mode_all_requires_yes_all(tmp_path: Path, capsys):
    archive = tmp_path / "NAS results"
    _seed_convnext_2d_archive(archive)
    data_out = tmp_path / "compact"
    try:
        export_main(
            [
                "--archive-root",
                str(archive),
                "--output-root",
                str(data_out),
                "--models",
                "convnext",
                "--layouts",
                "2d",
                "--mode",
                "all",
            ]
        )
        raised = False
    except SystemExit:
        raised = True
    assert raised
    out = capsys.readouterr().out
    assert "TOTAL:" in out
