"""Flat job list for multi-GPU best_config_runs scheduling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from rsna2024_lumbar.best_config_runs.config import iter_entries, list_best_config_files, load_best_config
from rsna2024_lumbar.best_config_runs.paths import run_output_dir


@dataclass(frozen=True)
class PipelineJob:
    best_config: str
    condition: str
    repeat_index: int
    split_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineJob:
        return cls(
            best_config=str(data["best_config"]),
            condition=str(data["condition"]),
            repeat_index=int(data["repeat_index"]),
            split_seed=int(data["split_seed"]),
        )


def iter_pipeline_jobs(
    config_dir: Path,
    *,
    repeats: int,
    seeds: Sequence[int] | None,
    only: list[str] | None,
    output_base: Path,
    epochs: int,
    skip_completed: bool,
) -> Iterator[PipelineJob]:
    from rsna2024_lumbar.best_config_runs.runner import resolve_repeat_seeds, run_is_complete

    repeat_seeds = resolve_repeat_seeds(repeats, seeds)
    want = {s.removesuffix(".json") for s in only} if only else None

    for cfg_path in list_best_config_files(config_dir):
        if want is not None and cfg_path.stem not in want:
            continue
        config = load_best_config(cfg_path)
        for entry in iter_entries(config):
            condition = str(entry["condition"])
            for repeat_index, split_seed in enumerate(repeat_seeds, start=1):
                run_dir = run_output_dir(
                    cfg_path.stem,
                    condition,
                    repeat_index,
                    output_base=output_base,
                )
                if skip_completed and run_is_complete(run_dir, epochs):
                    continue
                yield PipelineJob(
                    best_config=str(cfg_path.resolve()),
                    condition=condition,
                    repeat_index=repeat_index,
                    split_seed=int(split_seed),
                )


def list_pipeline_jobs(
    config_dir: Path,
    *,
    repeats: int,
    seeds: Sequence[int] | None,
    only: list[str] | None,
    output_base: Path,
    epochs: int,
    skip_completed: bool,
) -> list[PipelineJob]:
    return list(
        iter_pipeline_jobs(
            config_dir,
            repeats=repeats,
            seeds=seeds,
            only=only,
            output_base=output_base,
            epochs=epochs,
            skip_completed=skip_completed,
        )
    )


def write_jobs_file(jobs: list[PipelineJob], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [j.to_dict() for j in jobs]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_jobs_file(path: Path) -> list[PipelineJob]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [PipelineJob.from_dict(row) for row in data]
