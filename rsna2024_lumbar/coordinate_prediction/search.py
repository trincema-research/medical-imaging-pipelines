"""Landmark NAS grids: named families of 2D encoders × lr × weight decay."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rsna2024_lumbar.coordinate_prediction.constants import (
    BACKBONES,
    BATCH_SIZE_DEFAULT,
    CONDITIONS,
    IMAGE_SIZE_DEFAULT,
)

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_SPACE = CONFIGS_DIR / "landmark_nas_search_space.json"

FAMILY_FULL = "full"
FAMILY_CONVNEXT = "convnext"
FAMILY_EFFICIENTNET = "efficientnet"
FAMILY_LOCAL = "local"
FAMILIES = (FAMILY_FULL, FAMILY_CONVNEXT, FAMILY_EFFICIENTNET, FAMILY_LOCAL)
FAMILY_SPACES: dict[str, Path] = {
    FAMILY_FULL: DEFAULT_SPACE,
    FAMILY_CONVNEXT: CONFIGS_DIR / "landmark_nas_convnext.json",
    FAMILY_EFFICIENTNET: CONFIGS_DIR / "landmark_nas_efficientnet.json",
    FAMILY_LOCAL: CONFIGS_DIR / "landmark_nas_local.json",
}


@dataclass(frozen=True)
class LandmarkTrial:
    trial_id: int
    condition: str
    family: str
    backbone: str
    image_size: int
    batch_size: int
    lr: float
    weight_decay: float
    pretrained: bool
    epochs: int
    seed: int

    @property
    def slug(self) -> str:
        pre = "pt" if self.pretrained else "scratch"
        lr_s = f"{self.lr:.0e}".replace("-0", "-")
        wd_s = f"{self.weight_decay:g}".replace(".", "p")
        return (
            f"t{self.trial_id:03d}_{self.backbone}_sz{self.image_size}"
            f"_bs{self.batch_size}_lr{lr_s}_wd{wd_s}_{pre}"
        )


def load_space(path: Path | None = None) -> dict:
    src = Path(path) if path is not None else DEFAULT_SPACE
    return json.loads(src.read_text(encoding="utf-8"))


def resolve_space_path(*, family: str | None = None, space_path: Path | None = None) -> Path:
    if space_path is not None:
        return Path(space_path)
    key = family or FAMILY_FULL
    if key not in FAMILY_SPACES:
        raise KeyError(f"Unknown landmark NAS family {key!r}. Choose from {list(FAMILIES)}.")
    return FAMILY_SPACES[key]


def resolve_batch_size(space: dict, override: int | None = None) -> int:
    if override is not None:
        value = int(override)
        if value < 1:
            raise ValueError("--batch-size must be >= 1.")
        return value
    return int(space.get("batch_size") or BATCH_SIZE_DEFAULT)


def expand_trials(
    condition: str,
    *,
    family: str | None = None,
    space_path: Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> list[LandmarkTrial]:
    if condition not in CONDITIONS:
        raise KeyError(f"Unknown condition {condition!r}.")
    key = family or FAMILY_FULL
    space = load_space(resolve_space_path(family=key, space_path=space_path))
    backbones = list(space.get("backbones") or BACKBONES)
    sizes = space.get("image_sizes")
    if sizes is None:
        sizes = [space.get("image_size") or IMAGE_SIZE_DEFAULT]
    image_sizes = [int(v) for v in sizes]
    resolved_batch = resolve_batch_size(space, batch_size)
    lrs = [float(v) for v in space.get("learning_rates") or [1e-4]]
    wds = [float(v) for v in space.get("weight_decays") or [0.01]]
    pretrained_opts = [bool(v) for v in space.get("pretrained") or [True]]
    default_epochs = int(space.get("epochs") or 20)
    seed = int(space.get("seed") or 42)
    trials: list[LandmarkTrial] = []
    for idx, (backbone, image_size, lr, weight_decay, pretrained) in enumerate(
        itertools.product(backbones, image_sizes, lrs, wds, pretrained_opts), start=1
    ):
        trials.append(
            LandmarkTrial(
                trial_id=idx,
                condition=condition,
                family=key,
                backbone=str(backbone),
                image_size=int(image_size),
                batch_size=resolved_batch,
                lr=float(lr),
                weight_decay=float(weight_decay),
                pretrained=bool(pretrained),
                epochs=int(epochs) if epochs is not None else default_epochs,
                seed=seed,
            )
        )
    return trials


def trial_to_dict(trial: LandmarkTrial) -> dict:
    row = asdict(trial)
    row["slug"] = trial.slug
    return row
