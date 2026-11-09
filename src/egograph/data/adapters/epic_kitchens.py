"""EPIC-KITCHENS-100 adapter.

Layout under ``$EGO_DATA_ROOT/epic_kitchens``::

    EPIC-KITCHENS/P01/videos/P01_01.MP4        # 1080p, 59.94 fps GoPro (fisheye, no calibration)
    annotations/                               # git clone of epic-kitchens-100-annotations
        EPIC_100_train.csv, EPIC_100_validation.csv, EPIC_100_video_info.csv

The only labels are action segments (narration, verb, noun). There is no 3D ground truth:
EPIC is the in-the-wild input for the pose extractors run in Month 2.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path

from egograph.data.adapters.base import DatasetAdapter, register
from egograph.data.schema import ActionSegment, SequenceMeta
from egograph.data.video import probe


def _ts(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


@register
class EpicKitchensAdapter(DatasetAdapter):
    name = "epic_kitchens"
    display_name = "EPIC-KITCHENS-100"
    default_subdir = "epic_kitchens"
    description = "Unscripted kitchen activities, head-mounted GoPro, action narrations."

    @property
    def video_root(self) -> Path:
        return self.root / "EPIC-KITCHENS"

    @property
    def ann_root(self) -> Path:
        return self.root / "annotations"

    @cached_property
    def _video_info(self) -> dict[str, dict]:
        p = self.ann_root / "EPIC_100_video_info.csv"
        if not p.exists():
            return {}
        with p.open() as f:
            return {r["video_id"]: r for r in csv.DictReader(f)}

    @cached_property
    def _actions(self) -> dict[str, list[ActionSegment]]:
        out: dict[str, list[ActionSegment]] = defaultdict(list)
        for split in ("train", "validation"):
            p = self.ann_root / f"EPIC_100_{split}.csv"
            if not p.exists():
                continue
            with p.open() as f:
                for r in csv.DictReader(f):
                    out[r["video_id"]].append(ActionSegment(
                        start_s=_ts(r["start_timestamp"]), end_s=_ts(r["stop_timestamp"]),
                        label=r["narration"], verb=r["verb"], noun=r["noun"],
                        extra={"narration_id": r["narration_id"], "split": split,
                               "verb_class": int(r["verb_class"]), "noun_class": int(r["noun_class"])},
                    ))
        for v in out.values():
            v.sort(key=lambda a: a.start_s)
        return dict(out)

    def status(self) -> dict:
        st = super().status()
        vids = list(self.video_root.glob("P*/videos/*.MP4")) if self.video_root.exists() else []
        st.update(
            participants=sorted({p.parent.parent.name for p in vids}),
            videos=len(vids),
            partial=len(list(self.video_root.glob("P*/videos/*.aria2"))) if self.video_root.exists() else 0,
            annotations=self.ann_root.exists(),
        )
        return st

    def scan(self) -> Iterator[SequenceMeta]:
        for video in sorted(self.video_root.glob("P*/videos/*.MP4")):
            if video.with_name(video.name + ".aria2").exists():
                continue  # download still in progress
            vid = video.stem
            info = self._video_info.get(vid)
            if info:
                fps, dur = float(info["fps"]), float(info["duration"])
                w, h = (int(x) for x in info["resolution"].split("x"))
                n = int(round(fps * dur))
            else:
                vi = probe(video)
                fps, dur, w, h, n = vi.fps, vi.duration_s, vi.width, vi.height, vi.num_frames
            acts = self._actions.get(vid, [])
            splits = {a.extra["split"] for a in acts}
            yield SequenceMeta(
                dataset=self.name, sequence_id=vid,
                video_path=str(video.relative_to(self.root)),
                fps=fps, num_frames=n, width=w, height=h, duration_s=dur,
                has_actions=bool(acts),
                tags={"participant": vid.split("_")[0],
                      "split": splits.pop() if len(splits) == 1 else ("mixed" if splits else "test"),
                      "num_actions": len(acts),
                      "size_mb": round(video.stat().st_size / 2**20)},
            )

    def actions(self, meta: SequenceMeta) -> list[ActionSegment]:
        return self._actions.get(meta.sequence_id, [])
