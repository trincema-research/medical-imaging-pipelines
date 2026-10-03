"""Load a family JSON grid and expand TrialConfig rows (no GPU / no torch)."""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsna2024_lumbar.nas.families import FamilySpec, require_family
from rsna2024_lumbar.preprocessing.constants import (
    CONDITIONS,
    CROP_POLICY_CENTERED,
    png_size_for_policy,
)
from rsna2024_lumbar.preprocessing.crops import CropSettings, crop_settings_for


@dataclass
class SearchSpace:
    batch_sizes: list[int] = field(default_factory=lambda: [8])
    learning_rates: list[float] = field(default_factory=lambda: [3e-4, 1e-4, 5e-5])
    weight_decays: list[float] = field(default_factory=lambda: [5e-2, 1e-2])
    vit_variants: list[str] = field(
        default_factory=lambda: ["vit_b_16", "vit_l_16", "vit_b_32", "vit_l_32"]
    )
    maxvit_variants: list[str] = field(
        default_factory=lambda: ["maxvit_tiny_tf_224", "maxvit_small_tf_224"]
    )
    cnn2d_variants: list[str] = field(default_factory=lambda: ["convnext_small"])
    cnn3d_variants: list[str] = field(default_factory=lambda: ["cnn3d_s", "cnn3d_b"])
    effnet2d_variants: list[str] = field(default_factory=lambda: ["efficientnet_v2_s"])
    grids: list[dict[str, Any]] | None = None
    freeze_options: list[bool] = field(default_factory=lambda: [False])
    use_pretrained: list[bool] = field(default_factory=lambda: [True])
    amp_options: list[bool] = field(default_factory=lambda: [True])
    head_depths: list[int] = field(default_factory=lambda: [1])
    head_hidden_dims: list[int | None] = field(default_factory=lambda: [None])
    head_dropout_rates: list[float] = field(default_factory=lambda: [0.0])
    head_activations: list[str] = field(default_factory=lambda: ["gelu"])
    optimizer_types: list[str] = field(default_factory=lambda: ["AdamW"])
    scheduler_types: list[str] = field(default_factory=lambda: ["CosineAnnealingLR"])
    epochs: int = 50
    patience: int | None = None
    num_workers: int = 4
    search_mode: str = "grid"


@dataclass(frozen=True)
class TrialConfig:
    variant: str
    image_width: int
    image_height: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    freeze_backbone: bool
    use_pretrained: bool
    amp: bool
    head_depth: int
    head_hidden_dim: int | None
    head_dropout: float
    head_activation: str
    optimizer_type: str
    scheduler_type: str
    epochs: int
    patience: int
    num_workers: int
    seed: int
    condition: str
    crop_policy: str
    model_type: str
    input_layout: str
    crop_settings: CropSettings


def load_search_space(path: Path) -> SearchSpace:
    space = SearchSpace()
    overrides = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, value in overrides.items():
        if key == "image_sizes":
            continue
        if not hasattr(space, key):
            raise KeyError(f"Unknown search space key: {key}")
        setattr(space, key, value)
    return space


def backbone_variants(space: SearchSpace, spec: FamilySpec) -> list[str]:
    values = getattr(space, spec.variant_key)
    return list(values)


def build_trial_configs(
    space: SearchSpace,
    spec: FamilySpec,
    *,
    condition: str = "spinal_canal_stenosis",
    crop_policy: str = CROP_POLICY_CENTERED,
    base_seed: int = 1234,
) -> list[TrialConfig]:
    if condition not in CONDITIONS:
        raise KeyError(f"Unknown condition: {condition}")
    if space.grids:
        configs: list[TrialConfig] = []
        for grid in space.grids:
            sub = copy.deepcopy(space)
            sub.grids = None
            for key, value in grid.items():
                if not hasattr(sub, key):
                    raise KeyError(f"Unknown search space grid key: {key}")
                setattr(sub, key, value)
            configs.extend(
                build_trial_configs(
                    sub,
                    spec,
                    condition=condition,
                    crop_policy=crop_policy,
                    base_seed=base_seed + len(configs),
                )
            )
        return configs

    image_size = png_size_for_policy(crop_policy)
    settings = crop_settings_for(condition, crop_policy)
    patience = space.patience if space.patience is not None else space.epochs
    configs = []
    variants = backbone_variants(space, spec)
    for combo in itertools.product(
        variants,
        space.batch_sizes,
        space.learning_rates,
        space.weight_decays,
        space.freeze_options,
        space.use_pretrained,
        space.amp_options,
        space.head_depths,
        space.head_hidden_dims,
        space.head_dropout_rates,
        space.head_activations,
        space.optimizer_types,
        space.scheduler_types,
    ):
        (
            variant,
            batch_size,
            lr,
            wd,
            freeze,
            pretrained,
            amp,
            head_depth,
            head_hidden_dim,
            head_dropout,
            head_activation,
            optimizer_type,
            scheduler_type,
        ) = combo
        configs.append(
            TrialConfig(
                variant=variant,
                image_width=image_size.width,
                image_height=image_size.height,
                batch_size=batch_size,
                learning_rate=lr,
                weight_decay=wd,
                freeze_backbone=freeze,
                use_pretrained=pretrained,
                amp=amp,
                head_depth=head_depth,
                head_hidden_dim=head_hidden_dim,
                head_dropout=head_dropout,
                head_activation=head_activation,
                optimizer_type=optimizer_type,
                scheduler_type=scheduler_type,
                epochs=space.epochs,
                patience=patience,
                num_workers=space.num_workers,
                seed=base_seed,
                condition=condition,
                crop_policy=crop_policy,
                model_type=spec.model_type,
                input_layout=spec.input_layout,
                crop_settings=settings,
            )
        )
    return configs


def expand_family(
    family: str,
    *,
    condition: str = "spinal_canal_stenosis",
    crop_policy: str = CROP_POLICY_CENTERED,
) -> tuple[FamilySpec, SearchSpace, list[TrialConfig]]:
    spec = require_family(family)
    space = load_search_space(spec.config_file)
    trials = build_trial_configs(space, spec, condition=condition, crop_policy=crop_policy)
    return spec, space, trials
