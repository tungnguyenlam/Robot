"""Dataset adapter interface.

An adapter is the only code that knows a dataset's raw layout. It must provide:

* ``scan()``: walk the raw files and yield a ``SequenceMeta`` per video. This is slow
  and runs offline through ``egograph index``.
* ``video_path(meta)``: the absolute path to the video file.
* ``load_annotations(meta, frame_indices)``: an ``Annotations`` object for those frames,
  in the standard format.

To add a dataset, subclass ``DatasetAdapter``, decorate it with ``@register``, and import
it in ``egograph/data/adapters/__init__.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from egograph.data.schema import ActionSegment, Annotations, SequenceMeta
from egograph.paths import data_root

ADAPTERS: dict[str, type["DatasetAdapter"]] = {}


def register(cls: type["DatasetAdapter"]) -> type["DatasetAdapter"]:
    ADAPTERS[cls.name] = cls
    return cls


def get_adapter(name: str, **kwargs: Any) -> "DatasetAdapter":
    if name not in ADAPTERS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[name](**kwargs)


class DatasetAdapter(ABC):
    name: ClassVar[str]
    display_name: ClassVar[str]
    default_subdir: ClassVar[str]      # folder under EGO_DATA_ROOT
    description: ClassVar[str] = ""

    def __init__(self, root: str | Path | None = None, **options: Any):
        self.root = Path(root).expanduser() if root else data_root() / self.default_subdir
        self.options = options

    # ---- discovery -------------------------------------------------------------------
    def available(self) -> bool:
        return self.root.exists()

    @abstractmethod
    def scan(self) -> Iterator[SequenceMeta]:
        """Yield metadata for every complete, readable video under ``self.root``."""

    def status(self) -> dict[str, Any]:
        """Human-readable download state, shown in the viewer."""
        return {"available": self.available(), "root": str(self.root)}

    # ---- access ----------------------------------------------------------------------
    def video_path(self, meta: SequenceMeta) -> Path:
        return self.root / meta.video_path

    def actions(self, meta: SequenceMeta) -> list[ActionSegment]:
        return []

    def load_annotations(self, meta: SequenceMeta, frame_indices: np.ndarray | list[int]) -> Annotations:
        """Default implementation: no per-frame labels, only the action segments that
        overlap the window. Override this for datasets that have 3D annotations."""
        ann = Annotations.empty(np.asarray(frame_indices), meta.fps)
        ann.actions = overlapping(self.actions(meta), ann.timestamps)
        return ann


def overlapping(actions: list[ActionSegment], timestamps: np.ndarray) -> list[ActionSegment]:
    if len(timestamps) == 0:
        return []
    t0, t1 = float(np.min(timestamps)), float(np.max(timestamps))
    return [a for a in actions if a.end_s >= t0 and a.start_s <= t1]
