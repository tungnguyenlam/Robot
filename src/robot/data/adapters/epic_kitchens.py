"""EPIC-KITCHENS-100 adapter.

Layout under ``$EGO_DATA_ROOT/epic_kitchens``::

    EPIC-KITCHENS/P01/videos/P01_01.MP4        # 1080p, 59.94 fps GoPro (fisheye, no calibration)
    annotations/                               # git clone of epic-kitchens-100-annotations
        EPIC_100_train.csv, EPIC_100_validation.csv, EPIC_100_video_info.csv

The only labels are action segments (narration, verb, noun). There is no 3D ground truth:
EPIC is the in-the-wild input for the pose extractors run in Month 2.

Other modalities inside the MP4 files:
* audio: stereo AAC at 48 kHz in every video.
* imu: 142 of the 156 local videos (the GoPro Hero5/Hero7 recordings) carry a GPMF
  telemetry track with a ~200 Hz accelerometer and gyroscope (``robot.data.gpmf``).
  Parsing it reads the whole file, so the result is cached under
  ``$EGO_CACHE_ROOT/epic_kitchens/imu/<video>.npz`` (``robot prepare`` fills it ahead of time).

VISOR (``VISOR/``, see ``visor.py``): manual instance masks with hand contact on ~2 frames per
action, plus interpolated masks between them, for 62 of the local videos. Exposed as the
``seg_sparse`` modality through ``load_polygon_masks`` (cached like the IMU).
"""

from __future__ import annotations

import csv
import os
import threading
from collections import defaultdict
from collections.abc import Iterator
from functools import cached_property, lru_cache
from pathlib import Path

import av
import numpy as np

from robot.data.adapters import visor
from robot.data.adapters.base import DatasetAdapter, register
from robot.data.gpmf import read_imu, telemetry_stream
from robot.data.schema import ActionSegment, ImuTrack, PolygonMasks, SequenceMeta
from robot.data.video import probe
from robot.paths import cache_root


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

    @property
    def visor_root(self) -> Path:
        return self.root / "VISOR"

    @cached_property
    def _visor_videos(self) -> dict[str, str]:
        """Video id -> VISOR split, for videos with sparse or dense VISOR masks."""
        out: dict[str, str] = {}
        for p in sorted(self.visor_root.glob("Interpolations-DenseAnnotations/*/*_interpolations.zip")):
            out[p.name.removesuffix("_interpolations.zip")] = visor.split_of(p)
        for p in sorted(self.visor_root.glob("GroundTruth-SparseAnnotations/annotations/*/*.json")):
            out[p.stem] = visor.split_of(p)
        return out

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
            visor_split = self._visor_videos.get(vid)
            splits = {a.extra["split"] for a in acts}
            try:
                with av.open(str(video)) as c:
                    has_audio, has_imu = bool(c.streams.audio), telemetry_stream(c) is not None
                    codec = c.streams.video[0].codec_context.name
            except Exception:
                has_audio = has_imu = False
                codec = "?"
            yield SequenceMeta(
                dataset=self.name, sequence_id=vid,
                video_path=str(video.relative_to(self.root)),
                fps=fps, num_frames=n, width=w, height=h, duration_s=dur,
                has_actions=bool(acts),
                tags={"participant": vid.split("_")[0],
                      "split": splits.pop() if len(splits) == 1 else ("mixed" if splits else "test"),
                      "num_actions": len(acts), "codec": codec,
                      "size_mb": round(video.stat().st_size / 2**20),
                      **({"visor": visor_split} if visor_split else {})},
                modalities=["rgb", *(["audio"] if has_audio else []), *(["imu"] if has_imu else []),
                            *(["seg_sparse"] if visor_split else []), *(["actions"] if acts else [])],
            )

    def actions(self, meta: SequenceMeta) -> list[ActionSegment]:
        return self._actions.get(meta.sequence_id, [])

    def load_imu(self, meta: SequenceMeta) -> ImuTrack | None:
        if "imu" not in meta.modalities:
            return None
        cache = cache_root(self.name) / "imu" / f"{meta.sequence_id}.npz"
        if not cache.exists():
            d = read_imu(self.video_path(meta))
            if d is None:
                return None
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".{os.getpid()}-{threading.get_ident()}.tmp.npz")  # unique per writer
            np.savez_compressed(tmp, **d)
            tmp.replace(cache)
        with np.load(cache) as z:
            return ImuTrack(z["t"], z["accel"], z["gyro"], frame="camera")

    def load_polygon_masks(self, meta: SequenceMeta) -> PolygonMasks | None:
        if "seg_sparse" not in meta.modalities:
            return None
        return self._polygon_masks(meta.sequence_id, meta.video_path, meta.width, meta.height)

    @lru_cache(maxsize=4)  # noqa: B019  (one long-lived adapter per process)
    def _polygon_masks(self, sid: str, video_path: str, width: int, height: int) -> PolygonMasks | None:
        return visor.load(self.visor_root, cache_root(self.name) / "visor" / f"{sid}.npz", sid,
                          self.root / video_path, (width, height))
