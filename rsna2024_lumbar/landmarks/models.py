"""2D encoders with level-separated (x, y) heads.

Avengers (1st place): ConvNeXt-Base and EfficientNetV2-L, one 512×512 RGB
slice, five heads, L1 on (x/width, y/height).
"""

from __future__ import annotations

from rsna2024_lumbar.landmarks.constants import (
    BACKBONE_CONVNEXT_BASE,
    BACKBONE_CONVNEXT_TINY,
    BACKBONE_EFFNET_V2_L,
    BACKBONE_EFFNET_V2_S,
    LUMBAR_LEVELS,
)

_BUILDERS = {
    BACKBONE_CONVNEXT_TINY: ("convnext_tiny", "ConvNeXt_Tiny_Weights"),
    BACKBONE_CONVNEXT_BASE: ("convnext_base", "ConvNeXt_Base_Weights"),
    BACKBONE_EFFNET_V2_S: ("efficientnet_v2_s", "EfficientNet_V2_S_Weights"),
    BACKBONE_EFFNET_V2_L: ("efficientnet_v2_l", "EfficientNet_V2_L_Weights"),
}


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ImportError as exc:
        raise SystemExit("Landmark regression needs torch + torchvision. pip install -e '.[landmarks]'") from exc
    return torch, nn, models


def build_regressor(backbone: str, *, pretrained: bool = True, n_levels: int = len(LUMBAR_LEVELS)):
    """Return a 2D net: (B,3,H,W) → sigmoid xy (B,5,2) and vis logits (B,5)."""
    torch, nn, models = _require_torch()
    if backbone not in _BUILDERS:
        raise ValueError(f"Unknown backbone {backbone!r}. Choose from {sorted(_BUILDERS)}.")
    fn_name, weights_name = _BUILDERS[backbone]
    kwargs = {}
    if pretrained:
        weights_enum = getattr(models, weights_name, None)
        if weights_enum is not None:
            kwargs["weights"] = weights_enum.IMAGENET1K_V1
    encoder = getattr(models, fn_name)(**kwargs)
    last = encoder.classifier[-1]
    width = int(last.in_features)
    encoder.classifier = nn.Identity()

    class LevelHeadRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.xy_heads = nn.ModuleList([nn.Linear(width, 2) for _ in range(n_levels)])
            self.vis_heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(n_levels)])

        def forward(self, images):
            feats = self.encoder(images)
            if feats.ndim == 4:
                feats = feats.mean(dim=(2, 3))
            xy = torch.stack([head(feats) for head in self.xy_heads], dim=1)
            vis = torch.stack([head(feats).squeeze(-1) for head in self.vis_heads], dim=1)
            return torch.sigmoid(xy), vis

    return LevelHeadRegressor()


def landmark_loss(pred_xy, pred_vis, target_xy, visible, *, vis_weight: float = 0.25):
    torch, nn, _models = _require_torch()
    mask = visible.unsqueeze(-1)
    denom = mask.sum().clamp(min=1.0)
    loss_xy = ((pred_xy - target_xy).abs() * mask).sum() / denom
    loss_vis = nn.functional.binary_cross_entropy_with_logits(pred_vis, visible)
    return loss_xy + float(vis_weight) * loss_vis, {
        "l1": float(loss_xy.detach()),
        "bce": float(loss_vis.detach()),
    }


LEVEL_METRIC_KEYS = (
    "n_visible",
    "l1",
    "r2",
    "r2_x",
    "r2_y",
    "mae_px",
    "mae_x_px",
    "mae_y_px",
    "pck5",
    "pck10",
)


def empty_level_metrics() -> dict[str, float]:
    return {
        "n_visible": 0,
        "l1": float("nan"),
        "r2": float("nan"),
        "r2_x": float("nan"),
        "r2_y": float("nan"),
        "mae_px": float("nan"),
        "mae_x_px": float("nan"),
        "mae_y_px": float("nan"),
        "pck5": float("nan"),
        "pck10": float("nan"),
    }


def empty_coordinate_metrics(*, n_levels: int | None = None) -> dict:
    levels = list(LUMBAR_LEVELS if n_levels is None else LUMBAR_LEVELS[: int(n_levels)])
    out = empty_level_metrics()
    out["n"] = 0
    out["loss"] = float("nan")
    out["bce"] = float("nan")
    out["levels"] = {level: empty_level_metrics() for level in levels}
    for level in levels:
        for key in LEVEL_METRIC_KEYS:
            out[f"{key}_{level}"] = out["levels"][level][key]
    return out


def _r2_1d(pred, target) -> float:
    residual = ((pred - target) ** 2).sum()
    total = ((target - target.mean()) ** 2).sum()
    if float(total) <= 1e-12:
        return float("nan")
    return float(1.0 - residual / total)


def _r2_visible(pred_xy, target_xy, visible) -> dict[str, float]:
    mask = visible > 0.5
    empty = {"r2": float("nan"), "r2_x": float("nan"), "r2_y": float("nan")}
    if int(mask.sum()) < 2:
        return empty
    pred = pred_xy[mask]
    target = target_xy[mask]
    r2_x = _r2_1d(pred[:, 0], target[:, 0])
    r2_y = _r2_1d(pred[:, 1], target[:, 1])
    if r2_x != r2_x or r2_y != r2_y:
        r2 = float("nan")
    else:
        r2 = 0.5 * (r2_x + r2_y)
    return {"r2": r2, "r2_x": r2_x, "r2_y": r2_y}


def _visible_xy(pred_xy, target_xy, visible):
    if pred_xy.ndim == 3:
        pred_xy = pred_xy.reshape(-1, 2)
        target_xy = target_xy.reshape(-1, 2)
        visible = visible.reshape(-1)
    mask = visible > 0.5
    return pred_xy[mask], target_xy[mask], mask


def _errors_visible(pred_xy, target_xy, visible, width: int, height: int) -> dict[str, float]:
    torch, _nn, _models = _require_torch()
    pred, target, mask = _visible_xy(pred_xy, target_xy, visible)
    out = empty_level_metrics()
    out["n_visible"] = int(mask.sum())
    if out["n_visible"] == 0:
        return out
    dx = (pred[:, 0] - target[:, 0]) * float(width)
    dy = (pred[:, 1] - target[:, 1]) * float(height)
    dist = torch.sqrt(dx * dx + dy * dy)
    out["l1"] = float((pred - target).abs().mean().detach())
    out["mae_px"] = float(dist.mean().detach())
    out["mae_x_px"] = float(dx.abs().mean().detach())
    out["mae_y_px"] = float(dy.abs().mean().detach())
    out["pck5"] = float((dist <= 5.0).float().mean().detach())
    out["pck10"] = float((dist <= 10.0).float().mean().detach())
    if out["n_visible"] >= 2:
        out.update(_r2_visible(pred_xy, target_xy, visible))
    return out


def pixel_errors(pred_xy, target_xy, visible, width: int, height: int) -> dict[str, float]:
    return _errors_visible(pred_xy, target_xy, visible, width, height)


def coordinate_r2(pred_xy, target_xy, visible) -> dict[str, float]:
    """R² of visible (x/W, y/H): overall, per-axis, and per lumbar level."""
    out = _r2_visible(pred_xy, target_xy, visible)
    if getattr(pred_xy, "ndim", 0) == 3:
        n_levels = int(pred_xy.shape[1])
        for i, level in enumerate(LUMBAR_LEVELS[:n_levels]):
            part = _r2_visible(pred_xy[:, i], target_xy[:, i], visible[:, i])
            out[f"r2_{level}"] = part["r2"]
            out[f"r2_{level}_x"] = part["r2_x"]
            out[f"r2_{level}_y"] = part["r2_y"]
    return out


def coordinate_metrics(pred_xy, target_xy, visible, width: int, height: int) -> dict:
    """Overall + per-level R², pixel error, PCK, L1, and visible counts."""
    n_levels = int(pred_xy.shape[1]) if getattr(pred_xy, "ndim", 0) == 3 else 0
    out = _errors_visible(pred_xy, target_xy, visible, width, height)
    out["n"] = int(pred_xy.shape[0])
    levels: dict[str, dict[str, float]] = {}
    if n_levels:
        for i, level in enumerate(LUMBAR_LEVELS[:n_levels]):
            part = _errors_visible(pred_xy[:, i], target_xy[:, i], visible[:, i], width, height)
            levels[level] = part
            for key in LEVEL_METRIC_KEYS:
                out[f"{key}_{level}"] = part[key]
    out["levels"] = levels
    return out
