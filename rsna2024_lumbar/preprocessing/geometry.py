"""Patient-space geometry for intersecting sagittal and axial MRI planes.

DICOM LPS: +X left, +Y posterior, +Z superior. Kaggle (x, y) is (column, row).
PixelSpacing is (row spacing, column spacing). ImageOrientationPatient is
row-direction (increasing column) then column-direction (increasing row).
"""

from __future__ import annotations

import numpy as np

LPS_AXES: tuple[str, ...] = ("L", "P", "S")
PLANE_BY_NORMAL_AXIS: tuple[str, ...] = ("sagittal", "coronal", "axial")
PLANES: tuple[str, ...] = ("sagittal", "axial", "coronal", "oblique")


def pixel_to_patient(
    ipp: np.ndarray,
    iop: np.ndarray,
    pixel_spacing: tuple[float, float],
    x: float,
    y: float,
) -> np.ndarray:
    """Map a pixel (column x, row y) to LPS millimetres."""
    ipp = np.asarray(ipp, dtype=np.float64).reshape(3)
    iop = np.asarray(iop, dtype=np.float64).reshape(6)
    row_mm, col_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    return ipp + float(x) * col_mm * iop[:3] + float(y) * row_mm * iop[3:]


def patient_to_pixel(
    ipp: np.ndarray,
    iop: np.ndarray,
    pixel_spacing: tuple[float, float],
    xyz: np.ndarray,
) -> tuple[float, float]:
    """Project an LPS point onto a slice; return (column x, row y)."""
    ipp = np.asarray(ipp, dtype=np.float64).reshape(3)
    iop = np.asarray(iop, dtype=np.float64).reshape(6)
    point = project_to_plane(xyz, ipp, iop)
    delta = point - ipp
    row_mm, col_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    x = float(np.dot(delta, iop[:3]) / col_mm)
    y = float(np.dot(delta, iop[3:]) / row_mm)
    return x, y


def project_to_plane(xyz: np.ndarray, ipp: np.ndarray, iop: np.ndarray) -> np.ndarray:
    """Drop the point onto the slice plane along the slice normal."""
    ipp = np.asarray(ipp, dtype=np.float64).reshape(3)
    point = np.asarray(xyz, dtype=np.float64).reshape(3)
    normal = slice_normal(iop)
    return point - float(np.dot(point - ipp, normal)) * normal


def plane_offset_mm(ipp: np.ndarray, iop: np.ndarray, xyz: np.ndarray) -> float:
    """Signed millimetres from the slice plane to ``xyz`` along the normal."""
    ipp = np.asarray(ipp, dtype=np.float64).reshape(3)
    point = np.asarray(xyz, dtype=np.float64).reshape(3)
    return float(np.dot(point - ipp, slice_normal(iop)))


def interslice_spacing_mm(ipps: np.ndarray, iop: np.ndarray) -> float:
    """Median |Δ| along the slice normal for a stack of ImagePositionPatient values."""
    pts = np.asarray(ipps, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return 0.0
    along = pts @ slice_normal(iop)
    diffs = np.diff(np.sort(along))
    diffs = diffs[np.abs(diffs) > 1e-6]
    if diffs.size == 0:
        return 0.0
    return float(np.median(np.abs(diffs)))


def crop_around_pixel(
    gray: np.ndarray,
    xyz: np.ndarray,
    x: float,
    y: float,
    half_mm: float,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Crop a millimetre window around pixel (x, y). ``xyz`` is (H, W, 3)."""
    return crop_around_pixels(gray, xyz, [x], [y], half_mm)


def crop_around_pixels(
    gray: np.ndarray,
    xyz: np.ndarray,
    xs: list[float] | np.ndarray,
    ys: list[float] | np.ndarray,
    half_mm: float,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Crop a box covering every (x, y) plus a millimetre margin."""
    h, w = gray.shape[:2]
    if h < 2 or w < 2 or len(xs) == 0 or len(ys) == 0:
        return None
    dx = float(np.linalg.norm(xyz[0, 1] - xyz[0, 0]))
    dy = float(np.linalg.norm(xyz[1, 0] - xyz[0, 0]))
    hx = max(4, int(round(float(half_mm) / max(dx, 1e-6))))
    hy = max(4, int(round(float(half_mm) / max(dy, 1e-6))))
    x0 = max(0, int(np.floor(np.min(xs))) - hx)
    x1 = min(w, int(np.ceil(np.max(xs))) + hx + 1)
    y0 = max(0, int(np.floor(np.min(ys))) - hy)
    y1 = min(h, int(np.ceil(np.max(ys))) + hy + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return gray[y0:y1, x0:x1], xyz[y0:y1, x0:x1], x0, y0


def pixels_to_patient_grid(
    ipp: np.ndarray,
    iop: np.ndarray,
    pixel_spacing: tuple[float, float],
    height: int,
    width: int,
) -> np.ndarray:
    """LPS coordinates for every pixel. Shape (H, W, 3)."""
    ipp = np.asarray(ipp, dtype=np.float64).reshape(3)
    iop = np.asarray(iop, dtype=np.float64).reshape(6)
    row_mm, col_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    cols = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    return (
        ipp
        + cols[None, :, None] * col_mm * iop[:3]
        + rows[:, None, None] * row_mm * iop[3:]
    )


def slice_normal(iop: np.ndarray) -> np.ndarray:
    iop = np.asarray(iop, dtype=np.float64).reshape(6)
    normal = np.cross(iop[:3], iop[3:])
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return np.zeros(3, dtype=np.float64)
    return normal / norm


def classify_plane(iop: np.ndarray, *, threshold: float = 0.7) -> str:
    """Sagittal / coronal / axial from the slice normal, else oblique."""
    normal = np.abs(slice_normal(iop))
    axis = int(np.argmax(normal))
    if float(normal[axis]) < threshold:
        return "oblique"
    return PLANE_BY_NORMAL_AXIS[axis]


def plane_from_description(text: str) -> str | None:
    lowered = str(text).lower()
    if "sagittal" in lowered:
        return "sagittal"
    if "axial" in lowered:
        return "axial"
    if "coronal" in lowered:
        return "coronal"
    return None


def resolve_plane(description: str, iop: np.ndarray) -> str:
    """Prefer DICOM orientation; use the series name only if geometry is oblique."""
    geometric = classify_plane(iop)
    if geometric != "oblique":
        return geometric
    return plane_from_description(description) or "oblique"


def auto_voxel_mm(
    xyz: np.ndarray,
    cube_size: int,
    margin_mm: float,
) -> float:
    """Isotropic spacing so landmarks + margin fit inside the cube."""
    if cube_size < 2:
        raise ValueError("cube_size must be >= 2.")
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    span = float(np.max(pts.max(axis=0) - pts.min(axis=0))) if pts.size else 0.0
    extent = max(span + 2.0 * float(margin_mm), 2.0 * float(margin_mm), 1.0)
    return extent / float(cube_size - 1)


def voxel_origin(center: np.ndarray, cube_size: int, voxel_mm: float) -> np.ndarray:
    """LPS of voxel (0, 0, 0) so ``center`` maps to the cube midpoint."""
    half = 0.5 * (int(cube_size) - 1) * float(voxel_mm)
    return np.asarray(center, dtype=np.float64).reshape(3) - half


def world_to_voxel(xyz: np.ndarray, origin: np.ndarray, voxel_mm: float) -> np.ndarray:
    """Continuous [i, j, k] = [L, P, S] indices. Round for nearest voxel."""
    return (np.asarray(xyz, dtype=np.float64).reshape(-1, 3) - np.asarray(origin, dtype=np.float64).reshape(3)) / float(
        voxel_mm
    )


def voxel_to_world(ijk: np.ndarray, origin: np.ndarray, voxel_mm: float) -> np.ndarray:
    return np.asarray(origin, dtype=np.float64).reshape(3) + np.asarray(ijk, dtype=np.float64).reshape(-1, 3) * float(
        voxel_mm
    )


def lps_affine(origin: np.ndarray, voxel_mm: float) -> list[list[float]]:
    """4×4 voxel (i, j, k)=(L, P, S) → LPS millimetres."""
    s = float(voxel_mm)
    ox, oy, oz = [float(v) for v in np.asarray(origin, dtype=np.float64).reshape(3)]
    return [
        [s, 0.0, 0.0, ox],
        [0.0, s, 0.0, oy],
        [0.0, 0.0, s, oz],
        [0.0, 0.0, 0.0, 1.0],
    ]
