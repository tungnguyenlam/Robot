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


# --------------------------------------------------------------------------- Ego4D key frames
# Ego4D FHO (fho_main.json) labels, per narrated action, the frame where the hand first touches
# the object ("contact_frame", on about half the actions) and the point of no return where the
# object's state changes ("pnr_frame", on all of them). These are real human labels, not a
# proxy: the only contact timing we have outside HOI4D. fho_oscc-pnr.json repeats the same PNR
# frames (98% exact match on our videos), so it is not loaded.
KEYFRAME_KINDS = ("contact_frame", "pnr_frame")


def fho_keyframes(ann: Annotations) -> tuple[np.ndarray, bool]:
    """((K, T) bool, labelled): ``out[k, t]`` is True when key frame ``KEYFRAME_KINDS[k]`` of an
    action overlapping the window falls on clip frame ``t``. A key frame lands on the nearest
    clip frame if it is within half a frame step of it, so sparse labels never move by more than
    the sampling allows. ``labelled`` is False when no overlapping action has key frames, i.e.
    all-False means "unknown" rather than "no contact"."""
    fi = np.asarray(ann.frame_indices, np.int64)
    out = np.zeros((len(KEYFRAME_KINDS), len(fi)), bool)
    labelled = False
    if not len(fi):
        return out, labelled
    step = float(np.median(np.diff(fi))) if len(fi) > 1 else 1.0
    for a in ann.actions:
        cf = (a.extra or {}).get("critical_frames") or {}
        if not cf:
            continue
        labelled = True
        for k, kind in enumerate(KEYFRAME_KINDS):
            f = cf.get(kind)
            if f is None:
                continue
            t = int(np.argmin(np.abs(fi - int(f))))
            if abs(int(fi[t]) - int(f)) <= step / 2:
                out[k, t] = True
    return out, labelled
