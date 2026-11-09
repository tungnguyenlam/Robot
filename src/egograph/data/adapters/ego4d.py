"""Ego4D adapter (videos downloaded with the Lab22 CLI, see docs/DATASETS_GUIDE.md).

Expected layout under ``$EGO_DATA_ROOT/ego4d``, matching ``lab22 download ego4d -o data/ego4d``::

    **/full_scale/<video_uid>.mp4   or   **/clips/<clip_uid>.mp4
    **/ego4d.json                   # optional metadata (scenarios, duration)

Any ``*.mp4`` under the root is accepted, so other layouts also work. Ego4D has no 3D
hand or object ground truth. FHO hand boxes can be added later in ``load_annotations``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from functools import cached_property

from egograph.data.adapters.base import DatasetAdapter, register
from egograph.data.schema import SequenceMeta
from egograph.data.video import probe


@register
class Ego4DAdapter(DatasetAdapter):
    name = "ego4d"
    display_name = "Ego4D"
    default_subdir = "ego4d"
    description = "Large-scale daily-life egocentric video (3,670 h); FHO benchmark for hand-object interaction."

    @cached_property
    def _video_meta(self) -> dict[str, dict]:
        for p in sorted(self.root.rglob("ego4d.json")):
            try:
                d = json.loads(p.read_text())
                return {v["video_uid"]: v for v in d.get("videos", [])}
            except (json.JSONDecodeError, KeyError):
                continue
        return {}

    def status(self) -> dict:
        st = super().status()
        st["videos"] = sum(1 for _ in self.root.rglob("*.mp4")) if self.available() else 0
        if not st["videos"]:
            st["note"] = "Not downloaded yet. See docs/DATASETS_GUIDE.md section 1 (Lab22 CLI, needs LAB22_API_KEY)."
        return st

    def scan(self) -> Iterator[SequenceMeta]:
        for video in sorted(self.root.rglob("*.mp4")):
            try:
                vi = probe(video)
            except Exception:  # partial or corrupt download
                continue
            uid = video.stem
            vm = self._video_meta.get(uid, {})
            yield SequenceMeta(
                dataset=self.name, sequence_id=uid,
                video_path=str(video.relative_to(self.root)),
                fps=vi.fps, num_frames=vi.num_frames, width=vi.width, height=vi.height,
                duration_s=vi.duration_s,
                tags={"part": video.parent.name,
                      "scenarios": vm.get("scenarios", []),
                      "size_mb": round(video.stat().st_size / 2**20)},
            )
