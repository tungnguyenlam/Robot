"""Hand-object contact derived from 3D labels.

WHOLE predicts a binary contact flag per hand and EgoGrasp adds a contact loss; the Month 3
graph needs contact edges. None of the datasets ships contact labels, so we derive a proxy
from what HOI4D has: 3D hand joints and oriented object boxes (``T_cam_obj``, ``dimensions``).

``contact_dist[s, o, t]`` is the smallest signed distance from any of the 21 joints of hand
``s`` to the surface of box ``o`` (negative inside the box). ``contact = dist < threshold``.
The default threshold (``DEFAULT_THRESHOLD_M``) was picked against HOI4D's own action labels;
see docs/MODALITIES.md. Boxes are amodal and joints lie inside the skin, so treat this as a
proxy, not ground truth.
"""

from __future__ import annotations

import numpy as np

from egograph.data.schema import HAND_SIDES, Annotations

DEFAULT_THRESHOLD_M = 0.01  # HOI4D: holding 98-99% < 1 cm, rest 0%, reach-out 26% (2026-11-16)


def box_signed_distance(points_obj: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    """Signed distance (..., ) from points in the box frame to an axis-aligned box centred at the
    origin with extents ``dimensions``. Negative inside."""
    q = np.abs(points_obj) - np.asarray(dimensions, np.float32) / 2
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return outside + inside


def hand_object_distance(ann: Annotations, max_objects: int | None = None) -> np.ndarray:
    """(2, O, T) float32 signed distances in metres, NaN where the hand or object is missing."""
    objs = ann.objects[:max_objects] if max_objects else ann.objects
    T = ann.num_frames
    out = np.full((len(HAND_SIDES), len(objs), T), np.nan, np.float32)
    for s, side in enumerate(HAND_SIDES):
        h = ann.hands[side]
        for o, ob in enumerate(objs):
            if not np.isfinite(ob.dimensions).all():
                continue
            ok = h.valid & ob.valid & np.isfinite(h.joints_3d).all(axis=(1, 2)) & np.isfinite(ob.T_cam_obj).all(axis=(1, 2))
            for t in np.flatnonzero(ok):
                R, p = ob.T_cam_obj[t, :3, :3], ob.T_cam_obj[t, :3, 3]
                local = (h.joints_3d[t] - p) @ R  # R^T (x - p), row-vector form
                out[s, o, t] = box_signed_distance(local, ob.dimensions).min()
    return out


def hand_object_contact(ann: Annotations, threshold: float = DEFAULT_THRESHOLD_M,
                        max_objects: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns ((2, O, T) bool contact, (2, O, T) float distance)."""
    d = hand_object_distance(ann, max_objects)
    return np.nan_to_num(d, nan=np.inf) < threshold, d
