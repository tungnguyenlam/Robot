"""Camera and rigid-transform helpers shared by adapters, transforms, and the viewer."""

from __future__ import annotations

import numpy as np


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project camera-frame points (..., 3) to pixels (..., 2). Points with z <= 0 become NaN."""
    p = np.asarray(points, dtype=np.float64)
    z = p[..., 2:3]
    z = np.where(z > 1e-6, z, np.nan)
    uv = (p[..., :2] / z) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv.astype(np.float32)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to points (..., 3)."""
    return points @ T[:3, :3].T + T[:3, 3]


def box_corners(dimensions: np.ndarray) -> np.ndarray:
    """The 8 corners (8, 3) of an axis-aligned box centred at the origin. ``dimensions`` gives
    the extents along x, y, and z. The ordering follows ``BOX_EDGES``."""
    dx, dy, dz = np.asarray(dimensions, dtype=np.float32) / 2
    return np.array([[sx * dx, sy * dy, sz * dz]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float32)


BOX_EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))


def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Intrinsics after resizing the image by (sx, sy). Uses the pixel-centre convention."""
    K2 = np.asarray(K, dtype=np.float32).copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = (K2[0, 2] + 0.5) * sx - 0.5
    K2[1, 2] = (K2[1, 2] + 0.5) * sy - 0.5
    return K2


def crop_intrinsics(K: np.ndarray, x0: float, y0: float) -> np.ndarray:
    K2 = np.asarray(K, dtype=np.float32).copy()
    K2[0, 2] -= x0
    K2[1, 2] -= y0
    return K2


def default_intrinsics(width: int, height: int, hfov_deg: float = 90.0) -> np.ndarray:
    """Pinhole guess for datasets without calibration. Real egocentric cameras (GoPro, Aria)
    are fisheye; treat this only as an initialization for SLAM or pose methods."""
    f = (width / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float32)
