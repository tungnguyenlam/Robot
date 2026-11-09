"""The unified loader: (dataset, sequence, frame indices) -> standard ``Clip``.

``sample_windows`` turns indexes into a deterministic list of clip windows, either with a
sliding window or centred on annotated actions. ``EgoLoader.load_window`` decodes one window.
The torch ``EgoClipDataset`` (in ``torch_dataset.py``) wraps both.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from egograph.data.adapters import DatasetAdapter, get_adapter
from egograph.data.index import SequenceIndex
from egograph.data.schema import Clip, SequenceMeta
from egograph.data.video import VideoReader


@dataclass(frozen=True)
class ClipWindow:
    dataset: str
    sequence_id: str
    frame_indices: tuple[int, ...]
    label: str | None = None


def frame_indices_at(meta: SequenceMeta, start_s: float, num: int, fps: float | None = None) -> np.ndarray:
    """``num`` source-frame indices starting at ``start_s``, resampled to ``fps`` (default:
    the native rate). Indices are clamped to the video length."""
    fps = fps or meta.fps
    t = start_s + np.arange(num) / fps
    idx = np.round(t * meta.fps).astype(np.int64)
    return np.clip(idx, 0, max(meta.num_frames - 1, 0))


def sample_windows(
    metas: Iterable[SequenceMeta],
    clip_len: int,
    fps: float | None = None,
    stride_s: float | None = None,
    mode: str = "sliding",
    adapters: dict[str, DatasetAdapter] | None = None,
) -> list[ClipWindow]:
    """Enumerate clip windows.

    mode="sliding": windows every ``stride_s`` seconds (default: back to back).
    mode="actions": one window centred on each annotated action segment, which focuses
                    sampling on hand-object interaction in long videos such as EPIC.
    """
    out: list[ClipWindow] = []
    adapters = adapters or {}
    for m in metas:
        f = fps or m.fps
        span = clip_len / f
        if mode == "actions":
            ad = adapters.get(m.dataset) or get_adapter(m.dataset)
            for a in ad.actions(m):
                start = max(0.0, (a.start_s + a.end_s) / 2 - span / 2)
                out.append(ClipWindow(m.dataset, m.sequence_id,
                                      tuple(frame_indices_at(m, start, clip_len, f).tolist()), a.label))
        else:
            step = stride_s or span
            t = 0.0
            while t + span <= m.duration_s + 1e-6:
                out.append(ClipWindow(m.dataset, m.sequence_id, tuple(frame_indices_at(m, t, clip_len, f).tolist())))
                t += step
    return out


@dataclass
class EgoLoader:
    """Loads clips from any registered dataset. Keeps one adapter and index per dataset."""

    transform: Callable[[Clip], Clip] | None = None
    adapter_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    _adapters: dict[str, DatasetAdapter] = field(default_factory=dict, init=False, repr=False)
    _indexes: dict[str, SequenceIndex] = field(default_factory=dict, init=False, repr=False)

    def adapter(self, dataset: str) -> DatasetAdapter:
        if dataset not in self._adapters:
            self._adapters[dataset] = get_adapter(dataset, **self.adapter_kwargs.get(dataset, {}))
        return self._adapters[dataset]

    def index(self, dataset: str) -> SequenceIndex:
        if dataset not in self._indexes:
            self._indexes[dataset] = SequenceIndex.load(dataset)
        return self._indexes[dataset]

    def meta(self, dataset: str, sequence_id: str) -> SequenceMeta:
        return self.index(dataset)[sequence_id]

    frames_at = staticmethod(frame_indices_at)

    def load_clip(self, dataset: str, sequence_id: str, frame_indices: Iterable[int],
                  with_frames: bool = True) -> Clip:
        meta = self.meta(dataset, sequence_id)
        ad = self.adapter(dataset)
        fi = np.asarray(list(frame_indices), dtype=np.int64)
        frames = None
        if with_frames:
            with VideoReader(ad.video_path(meta)) as vr:
                frames = vr.read(fi)
        clip = Clip(meta=meta, frames=frames, K=meta.K, annotations=ad.load_annotations(meta, fi))
        return self.transform(clip) if self.transform else clip

    def load_window(self, w: ClipWindow, with_frames: bool = True) -> Clip:
        return self.load_clip(w.dataset, w.sequence_id, w.frame_indices, with_frames)
