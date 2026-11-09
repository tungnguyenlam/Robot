"""Geometric clip transforms. Each one updates the frames together with everything in pixel
space (K, joints_2d, bbox_2d), so labels stay aligned. 3D quantities are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from egograph.data.geometry import crop_intrinsics, scale_intrinsics
from egograph.data.schema import Clip


def _resize_frames(frames: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR)) for f in frames])


def _map_pixels(clip: Clip, sx: float, sy: float, dx: float, dy: float) -> None:
    """Apply ``u' = (u + 0.5) * sx - 0.5 - dx`` (and the same for v) to every 2D annotation.
    This is the pixel-centre convention, the same one ``scale_intrinsics`` uses."""
    ann = clip.annotations
    s = np.array([sx, sy], np.float32)
    off = 0.5 * s - 0.5 - np.array([dx, dy], np.float32)
    for h in ann.hands.values():
        h.joints_2d = h.joints_2d * s + off
    for o in ann.objects:
        o.bbox_2d = o.bbox_2d * np.tile(s, 2) + np.tile(off, 2)


@dataclass
class Resize:
    """Resize so the short side equals ``short_side`` (keeps the aspect ratio)."""

    short_side: int

    def __call__(self, clip: Clip) -> Clip:
        H, W = (clip.frames.shape[1:3] if clip.frames is not None else (clip.meta.height, clip.meta.width))
        s = self.short_side / min(H, W)
        w, h = int(round(W * s)), int(round(H * s))
        sx, sy = w / W, h / H
        if clip.frames is not None:
            clip.frames = _resize_frames(clip.frames, w, h)
        if clip.K is not None:
            clip.K = scale_intrinsics(clip.K, sx, sy)
        _map_pixels(clip, sx, sy, 0.0, 0.0)
        return clip


@dataclass
class CenterCrop:
    width: int
    height: int

    def __call__(self, clip: Clip) -> Clip:
        H, W = clip.frames.shape[1:3]
        x0, y0 = max((W - self.width) // 2, 0), max((H - self.height) // 2, 0)
        clip.frames = clip.frames[:, y0:y0 + self.height, x0:x0 + self.width]
        if clip.K is not None:
            clip.K = crop_intrinsics(clip.K, x0, y0)
        _map_pixels(clip, 1.0, 1.0, x0, y0)
        return clip


@dataclass
class Compose:
    transforms: list

    def __call__(self, clip: Clip) -> Clip:
        for t in self.transforms:
            clip = t(clip)
        return clip


def build_transforms(cfg: dict | None) -> Compose:
    """Build from config, for example ``{resize: 512, center_crop: [512, 512]}``."""
    cfg = cfg or {}
    ts: list = []
    if cfg.get("resize"):
        ts.append(Resize(int(cfg["resize"])))
    if cfg.get("center_crop"):
        w, h = cfg["center_crop"]
        ts.append(CenterCrop(int(w), int(h)))
    return Compose(ts)
