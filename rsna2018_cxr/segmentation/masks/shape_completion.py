"""Fill opaque lung from the pieces a U-Net can still see.

Air-trained lung models follow dark parenchyma and stop at consolidation.
The chest does not. This step keeps a mask that already has both costophrenic
angles, and otherwise places a mean lung field — learned from complete Normal
masks — onto the visible apices and lateral walls. A missing side is mirrored
from the side that still has a base.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from rsna2018_cxr.segmentation.masks.components import label_components


@dataclass(frozen=True)
class LungTemplate:
    """Mean lung occupancy in normalized bounding-box coordinates."""

    field: np.ndarray
    aspect: float
    samples: int
    tall_aspect: float = 0.0

    @property
    def shape_mask(self) -> np.ndarray:
        return self.field >= 0.48

    @property
    def flat_base_aspect(self) -> float:
        """Taller than the median, used only when the base is a flat cut through opacity."""
        stretched = self.aspect * 1.12
        return stretched if self.tall_aspect <= 0 else max(self.tall_aspect, stretched)


def build_lung_template(mask_paths: list[Path], size: int = 256, limit: int = 180) -> LungTemplate:
    """Average complete Normal lung fields into one canonical shape."""
    accumulator = np.zeros((size, size), dtype=np.float32)
    aspects: list[float] = []
    used = 0
    for path in mask_paths:
        if used >= limit:
            break
        mask = _read_mask(path)
        if mask is None or not _template_candidate(mask):
            continue
        ys, xs = np.nonzero(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        height, width = y1 - y0, x1 - x0
        if width < 8 or height < 8:
            continue
        accumulator += _resize_float(mask[y0:y1, x0:x1], size, size)
        aspects.append(height / width)
        used += 1
    if used < 5:
        raise ValueError(f"Need at least 5 complete Normal masks to build a lung shape, found {used}.")
    field = accumulator / float(used)
    return LungTemplate(
        field=field,
        aspect=float(np.median(aspects)),
        tall_aspect=float(np.percentile(aspects, 75)),
        samples=used,
    )


def complete_lung_field(
    gray: np.ndarray,
    unet: np.ndarray,
    template: LungTemplate,
) -> tuple[np.ndarray, str]:
    """Return a full lung field and a short reason code.

    ``keep`` means the U-Net already shows both bases. ``mirror`` copies a
    reliable lung across the spine. ``extend`` keeps the visible walls and
    paints the mean shape below them. ``shape`` replaces fragments that are
    too small to define a border.
    """
    image = np.asarray(gray)
    if image.ndim == 3:
        image = image[..., 0]
    height, width = image.shape[:2]
    observed = np.asarray(unet) > 0
    patient = _patient_mask(image)
    pieces = _lung_pieces(observed)
    rib_x0, rib_x1, rib_axis = _ribcage(patient)
    if int(pieces.sum()) == 0:
        predicted = _paste_shape(template, _body_box(patient, rib_x0, rib_x1, template.flat_base_aspect), (height, width))
        return _guard(observed, _finish(predicted & patient), pieces), "shape"

    axis = _split_axis(pieces, rib_axis)
    left = pieces[:, :axis]
    right = pieces[:, axis:]
    left_gray = image[:, :axis]
    right_gray = image[:, axis:]
    left_area = int(left.sum())
    right_area = int(right.sum())
    missing_left = left_area < 0.22 * max(right_area, 1)
    missing_right = right_area < 0.22 * max(left_area, 1)
    x0, x1 = _field_width(pieces, missing_left, missing_right, axis, rib_x0, rib_x1)
    y_top = int(np.flatnonzero(pieces.any(axis=1)).min())
    left_lat = _lateralness(left, "left") or 0.0
    right_lat = _lateralness(right, "right") or 0.0
    if (
        not missing_left
        and not missing_right
        and float(pieces.mean()) >= 0.24
        and left_lat >= 0.80
        and right_lat >= 0.80
    ):
        return _finish(pieces), "keep"
    left_complete = (not missing_left) and _real_base(left, left_gray, "left")
    right_complete = (not missing_right) and _real_base(right, right_gray, "right")
    if left_complete and right_complete:
        return _finish(pieces), "keep"
    if missing_left and right_complete:
        return _guard(observed, _finish((_mirror(right, axis, width) | pieces) & patient), pieces), "mirror"
    if missing_right and left_complete:
        return _guard(observed, _finish((_mirror(left, axis, width, origin=0) | pieces) & patient), pieces), "mirror"

    if left_complete:
        y_base = _bottom(left) + 1
    elif right_complete:
        y_base = _bottom(right) + 1
    else:
        goal = template.aspect * 1.18
        y_base = y_top + max(4, int(round((x1 - x0) * goal)))
    y_base = min(height, _body_limit(patient, height), max(y_base, int(np.flatnonzero(pieces.any(axis=1)).max()) + 1))
    predicted = _paste_shape(template, (x0, y_top, x1, y_base), (height, width)) & patient
    if int(pieces.sum()) < 0.35 * max(int(predicted.sum()), 1):
        return _guard(observed, _finish(predicted), pieces), "shape"
    return _guard(observed, _finish(_extend_below(pieces, predicted)), pieces), "extend"


def _finish(mask: np.ndarray) -> np.ndarray:
    from rsna2018_cxr.segmentation.masks.postprocess import remove_small_islands

    return remove_small_islands(mask, min_area_fraction=0.002)


def _guard(observed: np.ndarray, completed: np.ndarray, pieces: np.ndarray) -> np.ndarray:
    """A completion that erases a large aerated lung is a failed fit. Keep the pieces."""
    observed_area = float(observed.mean())
    if observed_area > 0.12 and float(completed.mean()) < 0.55 * observed_area:
        return _finish(pieces if int(pieces.sum()) else observed)
    return completed


def _template_candidate(mask: np.ndarray) -> bool:
    area = float(mask.mean())
    if area < 0.16 or area > 0.46:
        return False
    height, width = mask.shape
    axis = _split_axis(mask, width // 2)
    left, right = mask[:, :axis], mask[:, axis:]
    if min(int(left.sum()), int(right.sum())) < 0.35 * max(int(left.sum()), int(right.sum()), 1):
        return False
    ys, xs = np.nonzero(mask)
    box_h = int(ys.max() - ys.min())
    box_w = int(xs.max() - xs.min())
    if box_w < 8 or box_h / box_w < 0.72:
        return False
    return _tapered(left) and _tapered(right)


def _real_base(side: np.ndarray, gray_side: np.ndarray, lateral: str) -> bool:
    """A finished base ends on the lateral chest wall and is not a flat cut."""
    del gray_side
    lateralness = _lateralness(side, lateral)
    ratio = _height_over_width(side)
    if lateralness is None or ratio is None:
        return False
    if lateralness < 0.84 or ratio < 1.15:
        return False
    # A short lung with a sharp lateral angle is already finished (high diaphragm, large abdomen).
    if ratio < 1.45 and lateralness >= 0.90:
        return True
    return ratio >= 1.80


def _lateralness(side: np.ndarray, lateral: str) -> float | None:
    """1 when the lowest pixels sit on the lateral chest wall, 0 when they sit on the mediastinum."""
    rows = np.flatnonzero(side.any(axis=1))
    cols = np.flatnonzero(side.any(axis=0))
    if rows.size < 12 or cols.size < 6:
        return None
    y0, y1 = int(rows.min()), int(rows.max())
    x0, x1 = int(cols[0]), int(cols[-1])
    span = y1 - y0
    if span < 12 or x1 <= x0:
        return None
    bottom_band = side[y1 - max(3, int(0.10 * span)) : y1 + 1]
    bottom_cols = np.flatnonzero(bottom_band.any(axis=0))
    if bottom_cols.size == 0:
        return None
    position = (float(bottom_cols.mean()) - x0) / float(x1 - x0)
    return position if lateral == "right" else 1.0 - position


def _height_over_width(side: np.ndarray) -> float | None:
    rows = np.flatnonzero(side.any(axis=1))
    cols = np.flatnonzero(side.any(axis=0))
    if rows.size == 0 or cols.size < 2:
        return None
    return float(int(rows.max()) - int(rows.min())) / float(int(cols[-1]) - int(cols[0]))


def _tapered(side: np.ndarray) -> bool:
    """A real base narrows into the costophrenic angle. A cut across opaque lung does not."""
    rows = np.flatnonzero(side.any(axis=1))
    if rows.size < 12:
        return False
    y0, y1 = int(rows.min()), int(rows.max())
    span = y1 - y0
    if span < 12:
        return False
    mid = side[y0 + int(0.30 * span) : y0 + int(0.62 * span)]
    bot = side[y1 - max(3, int(0.12 * span)) : y1 + 1]
    mid_w = int(mid.any(axis=0).sum()) if mid.size else 0
    bot_w = int(bot.any(axis=0).sum()) if bot.size else 0
    if mid_w < 6:
        return False
    return bot_w < 0.62 * mid_w


def _extend_below(pieces: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Keep the visible contour and add the shape only where that column has ended."""
    height, width = pieces.shape
    observed = np.full(width, -1, dtype=np.int32)
    occupied = np.flatnonzero(pieces.any(axis=0))
    for column in occupied:
        observed[column] = int(np.flatnonzero(pieces[:, column]).max())
    rows = np.arange(height, dtype=np.int32)[:, None]
    below = (observed[None, :] >= 0) & (rows >= observed[None, :])
    empty = observed[None, :] < 0
    return pieces | (predicted & (below | empty))


def _field_width(
    pieces: np.ndarray,
    missing_left: bool,
    missing_right: bool,
    axis: int,
    rib_x0: int,
    rib_x1: int,
) -> tuple[int, int]:
    _height, width = pieces.shape
    rib_w = max(1, rib_x1 - rib_x0)
    if missing_left and not missing_right:
        x0, x1 = _span_across_spine(_side_span(pieces[:, axis:], axis), visible="right")
    elif missing_right and not missing_left:
        x0, x1 = _span_across_spine(_side_span(pieces[:, :axis], 0), visible="left")
    else:
        rows = np.flatnonzero(pieces.any(axis=1))
        y0 = int(rows.min())
        cut = y0 + max(8, int(0.60 * max(1, int(rows.max()) - y0)))
        upper = pieces[:cut]
        cols = np.flatnonzero(upper.any(axis=0))
        if cols.size == 0:
            cols = np.flatnonzero(pieces.any(axis=0))
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
    usable_rib = rib_w < int(0.92 * width)
    if usable_rib and (x1 - x0) < 0.55 * rib_w:
        inset = int(0.07 * rib_w)
        x0, x1 = rib_x0 + inset, rib_x1 - inset
    x0 = int(np.clip(x0, 0, width - 4))
    x1 = int(np.clip(x1, x0 + 4, width))
    return x0, x1


def _side_span(side: np.ndarray, origin: int) -> tuple[int, int]:
    cols = np.flatnonzero(side.any(axis=0))
    if cols.size == 0:
        return origin, origin + 1
    return origin + int(cols[0]), origin + int(cols[-1]) + 1


def _span_across_spine(span: tuple[int, int], visible: str) -> tuple[int, int]:
    """Widen a one-lung span to both lungs, leaving a mediastinal gap."""
    start, end = span
    lung_w = max(1, end - start)
    gap = int(round(0.12 * lung_w))
    if visible == "right":
        return start - gap - lung_w, end
    return start, end + gap + lung_w


def _mirror(side: np.ndarray, axis: int, width: int, origin: int | None = None) -> np.ndarray:
    """Reflect one lung across ``axis`` onto a full-width canvas."""
    canvas = np.zeros((side.shape[0], width), dtype=bool)
    if origin is None:
        canvas[:, axis:] = side
        source = canvas
    else:
        canvas[:, : side.shape[1]] = side
        source = canvas
    reflected = np.zeros_like(source)
    ys, xs = np.nonzero(source)
    mirrored = (2 * axis - xs).astype(np.int32)
    valid = (mirrored >= 0) & (mirrored < width)
    reflected[ys[valid], mirrored[valid]] = True
    return reflected


def _paste_shape(
    template: LungTemplate,
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = box
    height, width = shape
    x0 = int(np.clip(x0, 0, width - 1))
    x1 = int(np.clip(x1, x0 + 1, width))
    y0 = int(np.clip(y0, 0, height - 1))
    y1 = int(np.clip(y1, y0 + 1, height))
    if (x1 - x0) < 4 or (y1 - y0) < 4:
        return np.zeros(shape, dtype=bool)
    patch = _resize_float(template.shape_mask, y1 - y0, x1 - x0) >= 0.5
    out = np.zeros(shape, dtype=bool)
    out[y0:y1, x0:x1] = patch
    return out


def _body_box(body: np.ndarray, rib_x0: int, rib_x1: int, aspect: float) -> tuple[int, int, int, int]:
    height = body.shape[0]
    rib_w = max(1, rib_x1 - rib_x0)
    inset = int(0.07 * rib_w)
    x0, x1 = rib_x0 + inset, rib_x1 - inset
    y0 = _apex_from_body(body)
    y1 = min(height, y0 + int(round((x1 - x0) * aspect)))
    return x0, y0, x1, y1


def _apex_from_body(body: np.ndarray) -> int:
    height, width = body.shape
    rib_x0, rib_x1, _axis = _ribcage(body)
    target = 0.72 * max(1, rib_x1 - rib_x0)
    for y in range(int(height * 0.02), int(height * 0.45)):
        if int(body[y].sum()) >= target:
            return y
    return int(height * 0.08)


def _body_limit(body: np.ndarray, height: int) -> int:
    rows = np.flatnonzero(body.any(axis=1))
    if rows.size == 0:
        return height
    return min(height, int(rows.max()) + 1)


def _lung_pieces(unet: np.ndarray) -> np.ndarray:
    """Drop leads and collimation blobs. Never delete a component by brightness."""
    region = np.asarray(unet) > 0
    labels, count = label_components(region)
    if count == 0:
        return region
    height, width = region.shape
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    main = int(np.argmax(counts))
    ys, xs = np.nonzero(labels == main)
    pad_y = int(0.08 * height)
    pad_x = int(0.08 * width)
    y0, y1 = int(ys.min()) - pad_y, int(ys.max()) + pad_y
    x0, x1 = int(xs.min()) - pad_x, int(xs.max()) + pad_x
    keep = np.zeros(count + 1, dtype=bool)
    minimum = max(1, int(0.0015 * height * width))
    largest = int(counts[main])
    for index in range(1, count + 1):
        area = int(counts[index])
        if area < minimum:
            continue
        if area >= 0.18 * largest:
            keep[index] = True
            continue
        component_y, component_x = np.nonzero(labels == index)
        cy = float(component_y.mean())
        cx = float(component_x.mean())
        if y0 <= cy <= y1 and x0 <= cx <= x1:
            keep[index] = True
    if not keep.any():
        keep[main] = True
    return keep[labels]


def _patient_mask(gray: np.ndarray) -> np.ndarray:
    """Patient interior, for black or white collimation. Full frame when the surround is ambiguous."""
    image = np.asarray(gray)
    height, width = image.shape[:2]
    corner = max(8, min(height, width) // 30)
    samples = np.concatenate(
        [
            image[:corner, :corner].ravel(),
            image[:corner, -corner:].ravel(),
            image[-corner:, :corner].ravel(),
            image[-corner:, -corner:].ravel(),
        ]
    )
    background = float(np.median(samples))
    like_background = np.abs(image.astype(np.int16) - background) <= 16
    if float(like_background.mean()) > 0.85:
        return np.ones((height, width), dtype=bool)
    labels, count = label_components(like_background)
    if count == 0:
        return np.ones((height, width), dtype=bool)
    border_ids = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    border_ids = border_ids[border_ids != 0]
    if border_ids.size == 0:
        return np.ones((height, width), dtype=bool)
    surround = np.isin(labels, border_ids)
    patient = ~surround
    try:
        from scipy.ndimage import binary_fill_holes

        patient = binary_fill_holes(patient)
    except ImportError:
        pass
    if float(patient.mean()) < 0.12:
        return np.ones((height, width), dtype=bool)
    return patient


def _ribcage(body: np.ndarray) -> tuple[int, int, int]:
    """Narrow thoracic span, below the shoulders and above the abdomen."""
    height, width = body.shape
    spans: list[tuple[int, int, int]] = []
    for y in range(int(height * 0.20), int(height * 0.48)):
        cols = np.flatnonzero(body[y])
        if cols.size < 8:
            continue
        spans.append((int(cols[-1] - cols[0]), int(cols[0]), int(cols[-1])))
    if not spans:
        return int(width * 0.12), int(width * 0.88), width // 2
    spans.sort()
    _span, x0, x1 = spans[len(spans) // 5]
    return x0, x1, (x0 + x1) // 2


def _split_axis(mask: np.ndarray, fallback: int) -> int:
    _height, width = mask.shape
    fallback = int(np.clip(fallback, 0, width - 1))
    column = mask.sum(axis=0)
    lo = max(0, fallback - width // 10)
    hi = min(width, fallback + width // 10)
    if int(column[lo:hi].sum()) == 0:
        return fallback
    return lo + int(np.argmin(column[lo:hi]))


def _bottom(side: np.ndarray) -> int | None:
    rows = np.flatnonzero(side.any(axis=1))
    if rows.size == 0:
        return None
    return int(rows.max())


def _resize_float(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    image = Image.fromarray(np.clip(np.asarray(mask, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8))
    resized = image.resize((width, height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def render_completed_inspection(
    manifest_csv: Path,
    overlay_dir: Path,
    *,
    normal_mask_dir: Path,
) -> LungTemplate:
    """Write green completed contours for the studies that already have an inspection overlay."""
    import pandas as pd

    from rsna2018_cxr.segmentation.data.images import read_grayscale
    from rsna2018_cxr.segmentation.masks.agreement import boundary

    frame = pd.read_csv(manifest_csv)
    chosen = frame.dropna(subset=["inspection_overlay"])
    normals = [
        path
        for path in sorted(normal_mask_dir.glob("*.png"))
        if "_classical" not in path.name and "_consensus" not in path.name
    ]
    template = build_lung_template(normals)
    print(
        f"template samples={template.samples} aspect={template.aspect:.3f} "
        f"flat={template.flat_base_aspect:.3f}"
    )
    for row in chosen.to_dict(orient="records"):
        gray = read_grayscale(Path(row["image_path"]))
        unet = _read_mask(Path(row["unet_mask_path"]))
        if unet is None:
            continue
        completed, reason = complete_lung_field(gray, unet, template)
        rgb = np.stack([gray, gray, gray], axis=-1)
        rgb[boundary(completed)] = (0, 255, 0)
        destination = overlay_dir / str(row["severity_class"]) / f"{row['patient_id']}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(destination)
        print(
            f"{row['severity_class']} {row['patient_id'][:8]} {reason} "
            f"area {completed.mean():.3f} was {float(unet.mean()):.3f}"
        )
    return template


def _read_mask(path: Path) -> np.ndarray | None:
    name = path.name
    if "_classical" in name or "_consensus" in name:
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0
