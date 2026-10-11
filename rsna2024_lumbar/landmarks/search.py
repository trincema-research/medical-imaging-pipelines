"""Landmark NAS grid: 2D backbone × size × batch × lr (no torch)."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rsna2024_lumbar.landmarks.constants import BACKBONES, CONDITIONS

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_SPACE = CONFIGS_DIR / "landmark_nas_search_space.json"


@dataclass(frozen=True)
class LandmarkTrial:
    trial_id: int
    condition: str
    backbone: str
    image_size: int
    batch_size: int
    lr: float
    pretrained: bool
    epochs: int
    seed: int

    @property
    def slug(self) -> str:
        pre = "pt" if self.pretrained else "scratch"
        lr_s = f"{self.lr:.0e}".replace("-0", "-")
        return (
            f"t{self.trial_id:03d}_{self.backbone}_sz{self.image_size}"
            f"_bs{self.batch_size}_lr{lr_s}_{pre}"
        )


def load_space(path: Path | None = None) -> dict:
    src = Path(path) if path is not None else DEFAULT_SPACE
    return json.loads(src.read_text(encoding="utf-8"))


def expand_trials(
    condition: str,
    *,
    space_path: Path | None = None,
    epochs: int | None = None,
) -> list[LandmarkTrial]:
    if condition not in CONDITIONS:
        raise KeyError(f"Unknown condition {condition!r}.")
    space = load_space(space_path)
    backbones = list(space.get("backbones") or BACKBONES)
    sizes = [int(v) for v in space.get("image_sizes") or [256]]
    batches = [int(v) for v in space.get("batch_sizes") or [8]]
    lrs = [float(v) for v in space.get("learning_rates") or [1e-4]]
    pretrained_opts = [bool(v) for v in space.get("pretrained") or [True]]
    default_epochs = int(space.get("epochs") or 20)
    seed = int(space.get("seed") or 42)
    trials: list[LandmarkTrial] = []
    combo = itertools.product(backbones, sizes, batches, lrs, pretrained_opts)
    for idx, (backbone, image_size, batch_size, lr, pretrained) in enumerate(combo, start=1):
        trials.append(
            LandmarkTrial(
                trial_id=idx,
                condition=condition,
                backbone=str(backbone),
                image_size=int(image_size),
                batch_size=int(batch_size),
                lr=float(lr),
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
