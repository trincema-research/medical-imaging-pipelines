"""NAS search spaces for RSNA 2024 lumbar (ViT, MaxViT, ConvNeXt, ConvNeXt3D, EfficientNet, EfficientNet3D)."""

from rsna2024_lumbar.nas.families import FAMILIES, FAMILY_SPECS, FamilySpec, require_family
from rsna2024_lumbar.nas.gpus import ALLOWED_NAS_GPU_COUNTS, GpuShard, resolve_num_gpus, shard_trial_range
from rsna2024_lumbar.nas.search_space import (
    SearchSpace,
    TrialConfig,
    build_trial_configs,
    expand_family,
    load_search_space,
)

__all__ = [
    "ALLOWED_NAS_GPU_COUNTS",
    "FAMILIES",
    "FAMILY_SPECS",
    "FamilySpec",
    "GpuShard",
    "SearchSpace",
    "TrialConfig",
    "build_trial_configs",
    "expand_family",
    "load_search_space",
    "require_family",
    "resolve_num_gpus",
    "shard_trial_range",
]
