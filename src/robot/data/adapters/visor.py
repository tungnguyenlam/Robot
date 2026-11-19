"""EPIC-KITCHENS VISOR masks (used by the EPIC adapter; not a dataset of its own).

Layout under ``$EGO_DATA_ROOT/epic_kitchens/VISOR`` (data.bris DOI 2v6cgv1x04ol22qp9rm9x2j6a7)::

    GroundTruth-SparseAnnotations/annotations/{train,val}/<video>.json
        manual masks on ~2 frames per action, polygons in 1920x1080 pixels; hand masks carry
        ``in_contact_object`` (a mask id in the same frame, or a reason for no id)
    Interpolations-DenseAnnotations/{train,val}/<video>_interpolations.zip
        one JSON (up to 1.4 GB) of masks interpolated between pairs of manual frames,
        polygons in 854x480 pixels; frames at both ends repeat the manual masks (``type`` 1)

Frame numbers. VISOR frame ``n`` (1-based, in the file names) was extracted by time at a
constant ``R = round(nominal fps)`` (60 for the 59.94 fps EPIC-55 videos, 50 for the 50 fps
ones): it is the last video frame whose timestamp is earlier than ``n / R`` seconds.
:func:`frame_map` finds that frame from the video's packet timestamps and returns the index
``VideoReader`` gives it. A closed formula (``ceil(n * fps / R) - 1``) is wrong by a frame
after a timestamp gap: P05_08 has one, and drifts after n ~ 35k. Checked by matching VISOR's own
JPEGs against decoded frames on P01_03, P05_08 (59.94 fps) and P01_103 (50 fps).
``frame_mapping.json`` maps to the older ``rgb_frames`` release instead and is not used.

Parsing the dense JSON is slow and memory-hungry, so each video is converted once into
``$EGO_CACHE_ROOT/epic_kitchens/visor/<video>.npz`` (``robot prepare`` builds them all).
"""

from __future__ import annotations

import json
import os
import re
import threading
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np

from robot.data.schema import (CONTACT_NA, CONTACT_NONE, CONTACT_UNSEGMENTED, CONTACT_UNSURE,
                                  PolygonMasks)

CACHE_VERSION = 3  # 2: frames deduplicated after mapping; 3: OpenCV simplification
SPARSE_SIZE = (1920, 1080)
DENSE_SIZE = (854, 480)
# Polygon simplification tolerance in source pixels: ~1.7 px at 1080p for both. Cuts dense
# points ~4x; the interpolated outlines are not that accurate anyway.
SPARSE_EPS, DENSE_EPS = 1.7, 0.75
_FRAME = re.compile(r"frame_(\d+)\.")
_CONTACT_WORDS = {"hand-not-in-contact": CONTACT_NONE, "none-of-the-above": CONTACT_UNSEGMENTED,
                  "inconclusive": CONTACT_UNSURE}
_build_lock = threading.Lock()  # one dense parse at a time: each can take several GB of RAM


def frame_map(video: Path, ns: np.ndarray) -> np.ndarray:
    """``VideoReader`` frame indices of VISOR frame numbers ``ns`` (see the module docstring)."""
    import av
    ns = np.asarray(ns, np.int64)
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        pts = np.sort(np.array([p.pts for p in c.demux(s) if p.pts is not None], np.int64))
        start = s.start_time or 0
        tb = Fraction(s.time_base)
        fps = float(s.average_rate or s.guessed_rate or 30)  # as VideoReader._frame_index
        R = round(float(s.base_rate or s.guessed_rate or fps))
    # last frame with (pts - start) * tb < n / R, in integers
    j = np.searchsorted((pts - start) * tb.numerator * R, ns * tb.denominator, side="left") - 1
    j = np.clip(j, 0, len(pts) - 1)
    return np.round((pts[j] - start) * float(tb) * fps).astype(np.int64)


def files(root: Path, video: str) -> tuple[Path | None, Path | None]:
    """(sparse json, dense zip) of one video, each None when absent."""
    sp = next(iter(sorted(root.glob(f"GroundTruth-SparseAnnotations/annotations/*/{video}.json"))), None)
    dn = next(iter(sorted(root.glob(f"Interpolations-DenseAnnotations/*/{video}_interpolations.zip"))), None)
    return sp, dn


def split_of(path: Path) -> str:
    return path.parent.name


def _frame_no(image: dict) -> int:
    return int(_FRAME.search(image["name"]).group(1))


try:  # optional: C implementation of the same algorithm, ~30x faster than simplify_np
    from cv2 import approxPolyDP as _cv_dp
except ImportError:  # pragma: no cover
    _cv_dp = None


def simplify(p: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker on a closed polygon (K,2): drop points within ``eps`` of the
    simplified outline. Keeps at least 3 points. Uses OpenCV when installed."""
    if len(p) <= 4:
        return p
    if _cv_dp is not None:
        q = _cv_dp(np.ascontiguousarray(p, np.float32).reshape(-1, 1, 2), eps, True).reshape(-1, 2)
        return q.astype(p.dtype) if len(q) >= 3 else p
    return simplify_np(p, eps)


def simplify_np(p: np.ndarray, eps: float) -> np.ndarray:
    """Pure-numpy :func:`simplify`."""
    if len(p) <= 4:
        return p
    far = int(np.argmax(((p - p[0]) ** 2).sum(1)))  # split the ring at the farthest point
    keep = np.zeros(len(p), bool)
    keep[[0, far]] = True
    stack = [(0, far), (far, len(p))]  # (i, j): open segment p[i] -> p[j % K]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = p[i], p[j % len(p)]
        seg = p[i + 1:j] - a
        ab = b - a
        L = float(np.hypot(*ab))
        d = np.abs(seg[:, 0] * ab[1] - seg[:, 1] * ab[0]) / L if L > 0 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    q = p[keep]
    return q if len(q) >= 3 else p


def _polys(segments: list, scale: tuple[float, float], eps: float) -> list[np.ndarray]:
    """Polygons with at least 3 points, simplified by ``eps`` source pixels, then scaled to the
    video resolution and rounded to int16 pixels."""
    out = []
    for s in segments:
        p = np.asarray(s, np.float64).reshape(-1, 2)
        if len(p) >= 3:
            out.append(np.round(simplify(p, eps) * scale).astype(np.int16))
    return out


def build(root: Path, video: str, video_path: Path, size: tuple[int, int]) -> PolygonMasks | None:
    """Parse one video's sparse and dense VISOR files. Manual frames win over interpolated ones
    on the same frame; interpolated frames shared by two interpolations are merged by name."""
    sp_path, dn_path = files(root, video)
    if sp_path is None and dn_path is None:
        return None
    names: dict[str, int] = {}
    ent = lambda n: names.setdefault(n, len(names))  # noqa: E731
    # frame number -> (manual, [(entity, contact, polys)])
    frames: dict[int, tuple[bool, list]] = {}
    if sp_path is not None:
        sx, sy = size[0] / SPARSE_SIZE[0], size[1] / SPARSE_SIZE[1]
        for im in json.loads(sp_path.read_text())["video_annotations"]:
            by_id = {a["id"]: a["name"] for a in im["annotations"]}
            masks = []
            for a in im["annotations"]:
                c = CONTACT_NA
                if "in_contact_object" in a:
                    v = a["in_contact_object"]
                    c = ent(by_id[v]) if v in by_id else _CONTACT_WORDS.get(v, CONTACT_UNSURE)
                polys = _polys(a["segments"], (sx, sy), SPARSE_EPS)
                if polys:
                    masks.append((ent(a["name"]), c, polys))
            frames[_frame_no(im["image"])] = (True, masks)
    if dn_path is not None:
        dx, dy = size[0] / DENSE_SIZE[0], size[1] / DENSE_SIZE[1]
        with _build_lock:
            with zipfile.ZipFile(dn_path) as z:
                dense = json.loads(z.read(z.namelist()[0]))["video_annotations"]
            for im in dense:
                n = _frame_no(im["image"])
                if n in frames and frames[n][0]:
                    continue  # manual annotation of this frame
                masks = frames.setdefault(n, (False, []))[1]
                have = {e for e, _, _ in masks}
                for a in im["annotations"]:
                    e = ent(a["name"])
                    polys = _polys(a["segments"], (dx, dy), DENSE_EPS)
                    if polys and e not in have:
                        masks.append((e, CONTACT_NA, polys))
                        have.add(e)
            del dense
    # Map to video frames. At 60 fps sampling of a 59.94 fps video (and across a timestamp gap)
    # two VISOR frames can land on the same video frame: keep the manual one, else the first.
    ns = sorted(n for n, (_, ms) in frames.items() if ms)
    chosen: dict[int, int] = {}
    for n, k in zip(ns, frame_map(video_path, np.array(ns, np.int64)).tolist() if ns else []):
        if k not in chosen or (frames[n][0] and not frames[chosen[k]][0]):
            chosen[k] = n
    ks = sorted(chosen)
    entity, contact, mask_start, poly_start, pts = [], [], [0], [0], []
    frame_start = [0]
    for k in ks:
        for e, c, polys in frames[chosen[k]][1]:
            entity.append(e); contact.append(c)
            for p in polys:
                pts.append(p); poly_start.append(poly_start[-1] + len(p))
            mask_start.append(mask_start[-1] + len(polys))
        frame_start.append(len(entity))
    return PolygonMasks(
        names=list(names), frames=np.array(ks, np.int64),
        manual=np.array([frames[chosen[k]][0] for k in ks], bool),
        frame_start=np.array(frame_start, np.int64), entity=np.array(entity, np.int32),
        contact=np.array(contact, np.int32), mask_start=np.array(mask_start, np.int64),
        poly_start=np.array(poly_start, np.int64),
        points=np.concatenate(pts) if pts else np.zeros((0, 2), np.int16))


def load(root: Path, cache: Path, video: str, video_path: Path, size: tuple[int, int]) -> PolygonMasks | None:
    """Cached :func:`build`. The cache is rebuilt when ``CACHE_VERSION`` changes."""
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if int(z["version"]) == CACHE_VERSION:
                return PolygonMasks(names=json.loads(str(z["names"])), **{
                    k: z[k] for k in ("frames", "manual", "frame_start", "entity", "contact",
                                      "mask_start", "poly_start", "points")})
    pm = build(root, video, video_path, size)
    if pm is None:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".{os.getpid()}-{threading.get_ident()}.tmp.npz")  # unique per writer
    np.savez_compressed(tmp, version=CACHE_VERSION, names=json.dumps(pm.names), frames=pm.frames,
                        manual=pm.manual, frame_start=pm.frame_start, entity=pm.entity,
                        contact=pm.contact, mask_start=pm.mask_start, poly_start=pm.poly_start,
                        points=pm.points)
    tmp.replace(cache)
    return pm
