import numpy as np
import pytest

from rsna2024_lumbar.preprocessing.geometry import (
    auto_voxel_mm,
    classify_plane,
    crop_around_pixels,
    interslice_spacing_mm,
    patient_to_pixel,
    pixel_to_patient,
    pixels_to_patient_grid,
    plane_offset_mm,
    project_to_plane,
    resolve_plane,
    voxel_origin,
    voxel_to_world,
    world_to_voxel,
)


SAG_IOP = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
AX_IOP = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def test_pixel_to_patient_planes_meet_at_origin():
    sag = pixel_to_patient((0.0, -32.0, 32.0), SAG_IOP, (1.0, 1.0), 32.0, 32.0)
    ax = pixel_to_patient((-32.0, -32.0, 0.0), AX_IOP, (1.0, 1.0), 32.0, 32.0)
    assert np.allclose(sag, (0.0, 0.0, 0.0), atol=1e-6)
    assert np.allclose(ax, (0.0, 0.0, 0.0), atol=1e-6)


def test_classify_plane_from_orientation():
    assert classify_plane(SAG_IOP) == "sagittal"
    assert classify_plane(AX_IOP) == "axial"
    assert resolve_plane("T2", AX_IOP) == "axial"
    assert resolve_plane("Sagittal T2", (0.7, 0.5, 0.5, 0.0, 0.7, 0.5)) in {
        "sagittal",
        "coronal",
        "axial",
        "oblique",
    }


def test_world_voxel_roundtrip_and_auto_spacing():
    xyz = np.array([[-16.0, 70.0, -500.0], [20.0, 90.0, -380.0]])
    spacing = auto_voxel_mm(xyz, 96, 16.0)
    assert spacing > 1.0
    origin = voxel_origin(xyz.mean(axis=0), 96, spacing)
    ijk = world_to_voxel(xyz, origin, spacing)
    back = voxel_to_world(ijk, origin, spacing)
    assert np.allclose(back, xyz)
    mid = world_to_voxel(xyz.mean(axis=0), origin, spacing)
    assert np.allclose(mid, (47.5, 47.5, 47.5))


def test_patient_to_pixel_roundtrip():
    ipp = (0.0, -32.0, 32.0)
    xyz = pixel_to_patient(ipp, SAG_IOP, (1.0, 1.0), 32.0, 40.0)
    x, y = patient_to_pixel(ipp, SAG_IOP, (1.0, 1.0), xyz)
    assert abs(x - 32.0) < 1e-6
    assert abs(y - 40.0) < 1e-6


def test_project_neighbor_keeps_inplane_xy():
    labeled_ipp = (0.0, -32.0, 32.0)
    neighbor_ipp = (4.5, -32.0, 32.0)
    xyz = pixel_to_patient(labeled_ipp, SAG_IOP, (1.0, 1.0), 32.0, 32.0)
    x, y = patient_to_pixel(neighbor_ipp, SAG_IOP, (1.0, 1.0), xyz)
    assert abs(x - 32.0) < 1e-6
    assert abs(y - 32.0) < 1e-6
    assert abs(plane_offset_mm(neighbor_ipp, SAG_IOP, xyz)) == pytest.approx(4.5, abs=1e-6)
    on_plane = project_to_plane(xyz, neighbor_ipp, SAG_IOP)
    assert abs(plane_offset_mm(neighbor_ipp, SAG_IOP, on_plane)) < 1e-6


def test_interslice_spacing_from_ipp():
    ipps = [(0.0, 0.0, 0.0), (0.0, 0.0, 4.0), (0.0, 0.0, 8.0)]
    assert interslice_spacing_mm(ipps, AX_IOP) == pytest.approx(4.0)


def test_crop_around_projected_marks():
    gray = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    xyz = pixels_to_patient_grid((0.0, 0.0, 0.0), AX_IOP, (1.0, 1.0), 64, 64)
    cropped = crop_around_pixels(gray, xyz, [32.0, 36.0], [20.0, 24.0], 4.0)
    assert cropped is not None
    patch, grid, x0, y0 = cropped
    assert patch.shape[0] < 64 and patch.shape[1] < 64
    assert patch.shape == grid.shape[:2]
    assert x0 >= 0 and y0 >= 0
