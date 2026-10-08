from __future__ import annotations

from pathlib import Path

from rsna2024_lumbar.best_config_runs.cloud_common import collect_pack_files


def test_collect_pack_includes_best_configs_and_code(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    year = repo / "rsna2024_lumbar"
    (year / "best_config_runs").mkdir(parents=True)
    (year / "best_config_runs" / "cli.py").write_text("# cli\n", encoding="utf-8")
    (year / "best_config_runs" / "spinal_level.py").write_text("# spinal\n", encoding="utf-8")
    (year / "preprocessing").mkdir()
    (year / "preprocessing" / "constants.py").write_text("# c\n", encoding="utf-8")
    (year / "nas").mkdir()
    (year / "nas" / "paths.py").write_text("# p\n", encoding="utf-8")
    (year / "__init__.py").write_text("", encoding="utf-8")
    raw = year / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "train.csv").write_text("x\n", encoding="utf-8")
    best = year / "data" / "nas_compact" / "best_configs"
    best.mkdir(parents=True)
    (best / "nas_best_vit_2d.json").write_text('{"entries":[]}', encoding="utf-8")
    (best / "nas_best_all.json").write_text('{"entries":[]}', encoding="utf-8")
    bundle = year / "nas" / "training_bundle"
    bundle.mkdir(parents=True)
    (bundle / "train_vit_lumbar.py").write_text("# train\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")

    import rsna2024_lumbar.nas.paths as nas_paths
    import rsna2024_lumbar.best_config_runs.cloud_common as cloud_common

    monkeypatch.setattr(nas_paths, "REPO_ROOT", repo)
    monkeypatch.setattr(nas_paths, "YEAR_ROOT", year)
    monkeypatch.setattr(cloud_common, "REPO_ROOT", repo)
    monkeypatch.setattr(cloud_common, "YEAR_ROOT", year)
    monkeypatch.setattr(cloud_common, "LABEL_CSVS", ("train.csv",))
    monkeypatch.setattr(cloud_common, "CODE_PATHS", (
        year / "best_config_runs",
        year / "preprocessing",
        year / "nas",
        year / "__init__.py",
        repo / "pyproject.toml",
    ))

    files = collect_pack_files(
        include_crops=False,
        crop_policy="centered",
        bundle_files=[bundle / "train_vit_lumbar.py"],
        best_config_dir=best,
    )
    names = {p.name for p in files}
    assert "nas_best_vit_2d.json" in names
    assert "nas_best_all.json" not in names
    assert "train_vit_lumbar.py" in names
    assert "cli.py" in names
    assert "spinal_level.py" in names
