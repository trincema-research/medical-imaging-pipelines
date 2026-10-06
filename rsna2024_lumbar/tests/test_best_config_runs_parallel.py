from __future__ import annotations

from rsna2024_lumbar.best_config_runs.jobs import PipelineJob
from rsna2024_lumbar.best_config_runs.parallel import shard_pipeline_jobs


def test_shard_pipeline_jobs_covers_all():
    jobs = [
        PipelineJob("a.json", "c1", i, 42)
        for i in range(1, 11)
    ]
    shards = shard_pipeline_jobs(jobs, 4)
    flat = [j for _, chunk in shards for j in chunk]
    assert len(flat) == 10
    assert {j.repeat_index for j in flat} == set(range(1, 11))
