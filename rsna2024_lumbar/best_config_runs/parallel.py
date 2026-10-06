"""Spawn one ``best_config_runs worker`` subprocess per GPU shard."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from rsna2024_lumbar.best_config_runs.jobs import PipelineJob, write_jobs_file
from rsna2024_lumbar.nas.gpus import GpuShard, resolve_num_gpus, shard_trial_range


def shard_pipeline_jobs(jobs: list[PipelineJob], num_gpus: int) -> list[tuple[GpuShard, list[PipelineJob]]]:
    if not jobs:
        return []
    shards = shard_trial_range(len(jobs), num_gpus)
    out: list[tuple[GpuShard, list[PipelineJob]]] = []
    for shard in shards:
        chunk = jobs[shard.start_trial - 1 : shard.end_trial]
        out.append((shard, chunk))
    return out


def spawn_worker(
    shard: GpuShard,
    jobs: list[PipelineJob],
    *,
    worker_argv_base: list[str],
    jobs_dir: Path | None = None,
) -> subprocess.Popen:
    if jobs_dir is None:
        jobs_dir = Path(tempfile.gettempdir())
    job_path = jobs_dir / f"best_config_runs_jobs_gpu{shard.gpu_id}.json"
    write_jobs_file(jobs, job_path)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(shard.gpu_id)
    env.setdefault("MPLBACKEND", "Agg")
    cmd = [
        sys.executable,
        "-m",
        "rsna2024_lumbar.best_config_runs",
        "worker",
        "--jobs-file",
        str(job_path),
        *worker_argv_base,
    ]
    print(f"GPU {shard.gpu_id}: {len(jobs)} job(s) -> {job_path.name}")
    return subprocess.Popen(cmd, env=env)


def run_parallel_workers(
    jobs: list[PipelineJob],
    *,
    num_gpus: int,
    visible_gpus: int | None,
    worker_argv_base: list[str],
    wait: bool = True,
) -> int:
    resolved = resolve_num_gpus(num_gpus, visible=visible_gpus)
    if resolved == 1 or len(jobs) <= 1:
        return -1  # caller runs sequential
    shards = shard_pipeline_jobs(jobs, resolved)
    with tempfile.TemporaryDirectory(prefix="best_config_runs_jobs_") as tmp:
        jobs_dir = Path(tmp)
        processes = [
            spawn_worker(shard, chunk, worker_argv_base=worker_argv_base, jobs_dir=jobs_dir)
            for shard, chunk in shards
            if chunk
        ]
        if not wait:
            return 0
        rc = 0
        for proc in processes:
            code = proc.wait()
            if code != 0:
                rc = code
        return rc
