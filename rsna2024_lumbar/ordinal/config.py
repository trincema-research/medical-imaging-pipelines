"""Load NAS best-config JSON/CSV entries for refit runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


def load_best_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if "entries" not in data:
            raise ValueError(f"Expected 'entries' in {path}")
        return data
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return {
            "schema_version": 1,
            "label": path.stem,
            "entries": df.to_dict(orient="records"),
        }
    raise ValueError(f"Unsupported best-config format: {path}")


def iter_entries(
    config: dict[str, Any],
    *,
    condition: str | None = None,
) -> Iterator[dict[str, Any]]:
    for entry in config.get("entries", []):
        if condition is not None and entry.get("condition") != condition:
            continue
        yield entry


def hyperparameters_dict(entry: dict[str, Any]) -> dict[str, Any]:
    hp = entry.get("hyperparameters")
    if isinstance(hp, dict):
        return hp
    return {k: v for k, v in entry.items() if k not in ("metrics", "condition", "family")}


def refit_slug(entry: dict[str, Any]) -> str:
    hp = hyperparameters_dict(entry)
    family = entry.get("family") or hp.get("model_type", "model")
    layout = entry.get("archive_layout") or hp.get("input_layout", "2d")
    condition = entry.get("condition", "unknown")
    return f"{family}_{layout}/{condition}"
