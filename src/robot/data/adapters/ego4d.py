"""Ego4D adapter (downloaded with the Lab22 CLI, see docs/DATASETS_GUIDE.md).

Layout under ``$EGO_DATA_ROOT/Ego4D`` (``lab22 download ego4d -o data/Ego4D``)::

    ego4d.json                          # metadata: fps, device, scenarios, has_imu, has_gaze, ...
    v2/full_scale/<video_uid>.mp4       # VP9 + (usually) AAC audio, 30 fps
    v2/imu/<video_uid>.csv              # component_idx, component_timestamp_ms, canonical_timestamp_ms,
                                        #   gyro_{x,y,z} (rad/s), accl_{x,y,z} (m/s^2)
    v2/gaze/<video_uid>.csv             # Pupil Labs gaze: canonical_timestamp_s, norm_pos_{x,y}, confidence
    v2/annotations/narration.json       # timestamped free-form narrations
    v2/annotations/fho_main.json        # FHO: narrated actions with pre/PNR/post frames and hand/object boxes

Timing: a full-scale video is the concatenation of its recording "components"; the
``canonical_*`` timestamps in the IMU and gaze CSVs are on the full video's timeline, so we
use them directly (rows without one are dropped).

The two big annotation files (1.1 GB and 2.1 GB of JSON) are too slow to parse per clip.
``robot index --dataset ego4d`` splits them into one small JSON per indexed video under
``$EGO_CACHE_ROOT/ego4d/ann/<video_uid>.json``, which ``load_annotations`` reads.

Conventions:
* IMU axes are the device's own frame (GoPro, Vuzix, ...), so ``ImuTrack.frame`` is "device".
* Gaze ``norm_pos`` follows Pupil Labs: origin at the bottom-left, y up. We convert to pixels
  with ``u = x * W``, ``v = (1 - y) * H``.
* There is no 3D hand or object ground truth.
"""

from __future__ import annotations

import ast
import csv
import json
from collections.abc import Iterator
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np

from robot.data.adapters.base import DatasetAdapter, overlapping, register
from robot.data.schema import ActionSegment, Annotations, ImuTrack, ObjectTrack, SequenceMeta
from robot.data.video import probe
from robot.paths import cache_root, data_root

ANN_CACHE_VERSION = 1


def _parse(v):
    """ego4d.json stores some nested values as Python-literal strings."""
    if isinstance(v, str) and (v.startswith(("[", "{")) or v in ("True", "False", "None")):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


@register
class Ego4DAdapter(DatasetAdapter):
    name = "ego4d"
    display_name = "Ego4D"
    default_subdir = "Ego4D"
    description = "Large-scale daily-life egocentric video with audio, IMU, gaze, narrations, and FHO hand/object boxes."

    def __init__(self, root=None, **options):
        if root is None:  # accept both spellings of the folder
            for sub in ("Ego4D", "ego4d"):
                if (data_root() / sub).exists():
                    root = data_root() / sub
                    break
        super().__init__(root, **options)

    @property
    def v2(self) -> Path:
        return self.root / "v2"

    @cached_property
    def _video_meta(self) -> dict[str, dict]:
        for p in [self.root / "ego4d.json", *sorted(self.root.rglob("ego4d.json"))]:
            if p.exists():
                try:
                    d = json.loads(p.read_text())
                    return {v["video_uid"]: {k: _parse(x) for k, x in v.items()} for v in d.get("videos", [])}
                except (json.JSONDecodeError, KeyError):
                    continue
        return {}

    def _videos(self) -> list[Path]:
        full = self.v2 / "full_scale"
        return sorted(full.glob("*.mp4")) if full.exists() else sorted(self.root.rglob("*.mp4"))

    def status(self) -> dict:
        st = super().status()
        st["videos"] = len(self._videos()) if self.available() else 0
        st["metadata"] = (self.root / "ego4d.json").exists()
        st["annotations"] = sorted(p.name for p in (self.v2 / "annotations").glob("*.json"))[:40] \
            if (self.v2 / "annotations").exists() else []
        if not st["videos"]:
            st["note"] = "Not downloaded yet. See docs/DATASETS_GUIDE.md section 1 (Lab22 CLI, needs LAB22_API_KEY)."
        return st

    # ---- discovery ---------------------------------------------------------------------
    def scan(self) -> Iterator[SequenceMeta]:
        metas = []
        for video in self._videos():
            try:
                vi = probe(video)
                import av
                with av.open(str(video)) as c:
                    has_audio = bool(c.streams.audio)
            except Exception:  # partial or corrupt download
                continue
            uid = video.stem
            vm = self._video_meta.get(uid, {})
            imu = (self.v2 / "imu" / f"{uid}.csv").exists()
            gaze = (self.v2 / "gaze" / f"{uid}.csv").exists()
            metas.append(SequenceMeta(
                dataset=self.name, sequence_id=uid,
                video_path=str(video.relative_to(self.root)),
                fps=vi.fps, num_frames=vi.num_frames, width=vi.width, height=vi.height,
                duration_s=vi.duration_s,
                tags={"part": video.parent.name,
                      "scenario": (vm.get("scenarios") or ["?"])[0],
                      "scenarios": vm.get("scenarios") or [],
                      "device": (vm.get("device") or "?").strip(),
                      "source": vm.get("video_source"),
                      "split_fho": vm.get("split_fho"),
                      "size_mb": round(video.stat().st_size / 2**20)},
                modalities=["rgb", *(["audio"] if has_audio else []), *(["imu"] if imu else []),
                            *(["gaze"] if gaze else [])],
            ))
        self._build_annotation_cache([m.sequence_id for m in metas])
        for m in metas:
            ann = self._ann_cache(m.sequence_id)
            m.has_actions = bool(ann["actions"])
            m.tags["num_actions"] = len(ann["actions"])
            m.tags["num_narrations"] = sum(1 for a in ann["actions"] if a["source"] == "narration")
            m.tags["num_fho"] = sum(1 for a in ann["actions"] if a["source"] == "fho")
            if ann["actions"]:
                m.modalities.append("actions")
            if any(a.get("boxes") for a in ann["actions"]):
                m.modalities += ["hand_box", "object_box"]
            if any(a.get("critical_frames") for a in ann["actions"]):
                m.modalities.append("keyframes")
                m.tags["num_contact_frames"] = sum(1 for a in ann["actions"]
                                                   if (a.get("critical_frames") or {}).get("contact_frame") is not None)
            yield m

    # ---- annotation cache --------------------------------------------------------------
    def _ann_path(self, uid: str) -> Path:
        return cache_root(self.name) / "ann" / f"{uid}.json"

    def _build_annotation_cache(self, uids: list[str]) -> None:
        """Split narration.json and fho_main.json into one file per video in ``uids``."""
        ann_dir = self.v2 / "annotations"
        want = set(uids)
        per: dict[str, list[dict]] = {u: [] for u in uids}
        stamp = {"version": ANN_CACHE_VERSION, "sources": {}}
        for name, fn in (("narration.json", _narrations), ("fho_main.json", _fho_actions)):
            p = ann_dir / name
            if not p.exists():
                continue
            stamp["sources"][name] = p.stat().st_size
            with p.open() as f:
                data = json.load(f)
            for uid, acts in fn(data, want):
                per[uid].extend(acts)
            del data
        out = cache_root(self.name) / "ann"
        out.mkdir(parents=True, exist_ok=True)
        for uid, acts in per.items():
            acts.sort(key=lambda a: a["start_s"])
            self._ann_path(uid).write_text(json.dumps({**stamp, "actions": acts}))
        _ann_cache_file.cache_clear()

    def _ann_cache(self, uid: str) -> dict:
        return _ann_cache_file(str(self._ann_path(uid)))

    # ---- labels ------------------------------------------------------------------------
    def actions(self, meta: SequenceMeta) -> list[ActionSegment]:
        return [ActionSegment(a["start_s"], a["end_s"], a["label"], a.get("verb"), a.get("noun"),
                              extra={k: v for k, v in a.items()
                                     if k not in ("start_s", "end_s", "label", "verb", "noun", "boxes")})
                for a in self._ann_cache(meta.sequence_id)["actions"]]

    def load_annotations(self, meta: SequenceMeta, frame_indices) -> Annotations:
        fi = np.asarray(frame_indices, dtype=np.int64)
        ann = Annotations.empty(fi, meta.fps)
        ann.actions = overlapping(self.actions(meta), ann.timestamps)
        # FHO boxes exist only on the pre/PNR/post key frames. Put each on the clip frame
        # whose source index matches exactly, so sparse labels never move in time.
        pos = {int(i): t for t, i in enumerate(fi)}
        objects: dict[str, ObjectTrack] = {}
        for a in self._ann_cache(meta.sequence_id)["actions"]:
            for b in a.get("boxes", []):
                t = pos.get(int(b["frame"]))
                if t is None:
                    continue
                if b["type"] in ("left_hand", "right_hand"):
                    h = ann.hands[b["type"].split("_")[0]]
                    h.bbox_2d[t] = b["xyxy"]
                else:  # object_of_change, tool
                    key = f'{b["type"]}:{b.get("noun") or "?"}'
                    if key not in objects:
                        objects[key] = ObjectTrack.empty(key, b.get("noun") or b["type"], len(fi))
                    o = objects[key]
                    o.bbox_2d[t] = b["xyxy"]
                    o.valid[t] = True  # 2D only: T_cam_obj stays NaN
        ann.objects = list(objects.values())
        ann.gaze_2d = self.load_gaze(meta, fi)
        return ann

    # ---- signals -----------------------------------------------------------------------
    def load_imu(self, meta: SequenceMeta) -> ImuTrack | None:
        if "imu" not in meta.modalities:
            return None
        d = _read_imu_csv(str(self.v2 / "imu" / f"{meta.sequence_id}.csv"))
        return ImuTrack(d["t"], d["accel"], d["gyro"], frame="device")

    def load_gaze(self, meta: SequenceMeta, frame_indices) -> np.ndarray | None:
        if "gaze" not in meta.modalities:
            return None
        t, xy = _read_gaze_csv(str(self.v2 / "gaze" / f"{meta.sequence_id}.csv"))
        ts = np.asarray(frame_indices, np.float64) / meta.fps
        out = np.full((len(ts), 2), np.nan, np.float32)
        if len(t) < 2:
            return out
        j = np.clip(np.searchsorted(t, ts), 1, len(t) - 1)
        near = np.minimum(np.abs(t[j] - ts), np.abs(t[j - 1] - ts)) < 0.1
        out[:, 0] = np.interp(ts, t, xy[:, 0]) * meta.width
        out[:, 1] = (1.0 - np.interp(ts, t, xy[:, 1])) * meta.height
        out[~near] = np.nan
        return out


# ---------------------------------------------------------------------------------- parsers
def _xyxy(box: dict) -> list[float]:
    x, y, w, h = (float(box[k]) for k in ("x", "y", "width", "height"))
    return [x, y, x + w, y + h]


def _narrations(data: dict, want: set[str]):
    """narration.json: {video_uid: {narration_pass_1: {narrations: [{timestamp_sec, narration_text, ...}]}}}.
    Narrations are instants; we give each a 1 s segment centred on it so window overlap works."""
    for uid in want & data.keys():
        acts = []
        for pass_name, block in (data[uid] or {}).items():
            if not isinstance(block, dict):
                continue
            for n in block.get("narrations") or []:
                t = float(n["timestamp_sec"])
                text = (n.get("narration_text") or "").replace("#C C ", "").replace("#c c ", "").strip()
                acts.append({"start_s": max(t - 0.5, 0.0), "end_s": t + 0.5, "label": text,
                             "source": "narration", "pass": pass_name, "timestamp_s": t,
                             "frame": n.get("timestamp_frame")})
        yield uid, acts


def _fho_actions(data: dict, want: set[str]):
    """fho_main.json: videos[].annotated_intervals[].narrated_actions[] with critical_frames and
    frames[].boxes[] (object_type in left_hand/right_hand/object_of_change/tool). ``frame_number``
    and ``critical_frames`` count frames of the full-scale video (checked: an action in an interval
    starting at frame 8999 has frame_number 8953 = clip_frame_number 194 + clip offset 8759), and
    boxes are xywh pixels at the full-scale resolution (max x+w, y+h = 1920, 1440 on a 1920x1440 video)."""
    for v in data.get("videos", []):
        uid = v.get("video_uid")
        if uid not in want:
            continue
        acts = []
        for iv in v.get("annotated_intervals") or []:
            for na in iv.get("narrated_actions") or []:
                # Rejected actions (e.g. "no hand-object interaction") carry no usable frames.
                if na.get("is_rejected") or not na.get("is_valid_action", True) or na.get("start_sec") is None:
                    continue
                boxes = []
                for fr in na.get("frames") or []:
                    for b in fr.get("boxes") or []:
                        bb = b.get("bbox")
                        if not bb:
                            continue
                        boxes.append({"frame": int(fr["frame_number"]), "frame_type": fr.get("frame_type"),
                                      "type": b.get("object_type"), "xyxy": _xyxy(bb),
                                      "noun": b.get("structured_noun")})
                cf = na.get("critical_frames") or {}
                acts.append({"start_s": float(na["start_sec"]), "end_s": float(na["end_sec"]),
                             "label": (na.get("narration_text") or "").replace("#C C ", "").strip(),
                             "verb": na.get("structured_verb") or na.get("freeform_verb"),
                             "noun": next((b["noun"] for b in boxes if b["type"] == "object_of_change" and b["noun"]),
                                          None),
                             "source": "fho", "clip_uid": iv.get("clip_uid"),
                             "state_transition": na.get("state_transition"),
                             # full-video frame numbers: pre_45/30/15, pre_frame, contact_frame, pnr_frame, post_frame
                             "critical_frames": {k: v for k, v in cf.items() if v is not None},
                             "boxes": boxes})
        yield uid, acts


@lru_cache(maxsize=512)
def _ann_cache_file(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {"actions": []}


@lru_cache(maxsize=16)
def _read_imu_csv(path: str) -> dict[str, np.ndarray]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r["canonical_timestamp_ms"]:
                continue
            try:
                rows.append([float(r["canonical_timestamp_ms"]) / 1000.0,
                             *(float(r[k] or "nan") for k in ("accl_x", "accl_y", "accl_z", "gyro_x", "gyro_y", "gyro_z"))])
            except ValueError:
                continue
    a = np.asarray(rows, np.float64).reshape(-1, 7)
    a = a[np.argsort(a[:, 0], kind="stable")]  # some files have non-monotonic rows
    return {"t": a[:, 0], "accel": a[:, 1:4].astype(np.float32), "gyro": a[:, 4:7].astype(np.float32)}


@lru_cache(maxsize=16)
def _read_gaze_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    t, xy = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                ts, x, y = float(r["canonical_timestamp_s"]), float(r["norm_pos_x"]), float(r["norm_pos_y"])
            except (ValueError, KeyError):
                continue
            if float(r.get("confidence") or 1.0) < 0.6:
                continue
            t.append(ts)
            xy.append((x, y))
    t, xy = np.asarray(t, np.float64), np.asarray(xy, np.float32).reshape(-1, 2)
    o = np.argsort(t, kind="stable")
    return t[o], xy[o]
