"""HOI4D adapter, the 3D ground-truth source (MANO hands, 6-DoF objects, camera trajectory).

Layout under ``$EGO_DATA_ROOT/HOI4D`` (as downloaded and extracted on amdserver)::

    HOI4D_release/<seq>/align_rgb/image.mp4          # 1920x1080, 15 fps, ~300 frames
    HOI4D_annotations/<seq>/objpose/<frame>.json     # per-frame 6-DoF object box(es)
    HOI4D_annotations/<seq>/action/color.json        # action events (seconds)
    HOI4D_annotations/<seq>/3Dseg/output.log         # camera trajectory (Open3D .log)
    Hand_pose/handpose_{left,right}_hand/<seq>/<frame>.pickle   # MANO poseCoeff/beta/trans/kps2D
    camera_params/<ZY...>/intrin.npy                 # 3x3 K per capture device

``<seq>`` is ``ZY2021080000x/H?/C?/N?/S?/s?/T?``: device / human / category / object
instance / scene / layout / task. The sequence id replaces '/' with '-'.

Conventions, checked against the data with tests/test_hoi4d_conventions.py (2026-11-09):
* objpose ``rotation`` holds intrinsic XYZ Euler angles in radians, ``center`` is in the
  camera frame in meters, and ``dimensions`` (length, width, height) are the extents along
  the object x, y, and z axes. The projected 3D box matches the annotated 2D box with mean
  IoU 0.945 over 140 samples; the next-best convention scores 0.71.
* hand ``poseCoeff`` is 48-D axis-angle (no PCA, flat_hand_mean=True) and ``trans`` is in
  the camera frame. ``kps2D`` is in the 21-joint OpenPose order. Projected MANO joints are
  0.7 px (right) and 0.85 px (left) from kps2D (median). With flat_hand_mean=False the error
  is 32 px; with the smplx left-shapedirs fix it is 29 px.
"""

from __future__ import annotations

import json
import os
import pickle
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from egograph.data.adapters.base import DatasetAdapter, overlapping, register
from egograph.data.geometry import make_T
from egograph.data.schema import ActionSegment, Annotations, ObjectTrack, SequenceMeta
from egograph.data.video import probe

CATEGORIES = {
    "C1": "ToyCar", "C2": "Mug", "C3": "Laptop", "C4": "StorageFurniture", "C5": "Bottle",
    "C6": "Safe", "C7": "Bowl", "C8": "Bucket", "C9": "Scissors", "C11": "Pliers",
    "C12": "Kettle", "C13": "Knife", "C14": "TrashCan", "C17": "Lamp", "C18": "Stapler",
    "C20": "Chair",
}
RIGID = {"ToyCar", "Mug", "Bottle", "Bowl", "Kettle", "Knife"}

HOI4D_FPS = 15.0


def parse_open3d_log(path: Path) -> dict[int, np.ndarray]:
    """Parse an Open3D trajectory .log file into {frame: 4x4 T_world_cam}."""
    poses: dict[int, np.ndarray] = {}
    lines = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    for i in range(0, len(lines) - 4, 5):
        frame = int(lines[i][0])
        poses[frame] = np.array([[float(x) for x in row] for row in lines[i + 1:i + 5]], dtype=np.float32)
    return poses


@register
class HOI4DAdapter(DatasetAdapter):
    name = "hoi4d"
    display_name = "HOI4D"
    default_subdir = "HOI4D"
    description = "Egocentric RGB-D (head-mounted Kinect) with MANO hands, 6-DoF object poses, and camera trajectories."

    def __init__(self, root=None, mano: bool | None = None, **options):
        super().__init__(root, **options)
        # Running MANO forward kinematics gives joints_3d. It needs torch, smplx, and the
        # MANO pickles. The default is to use it when those are importable and present.
        self._want_mano = mano
        self._mano = None

    # ---- paths -------------------------------------------------------------------------
    @property
    def release_root(self) -> Path:
        return self.root / "HOI4D_release"

    def _rel(self, meta: SequenceMeta) -> str:
        return meta.tags["rel"]

    def _ann(self, meta: SequenceMeta) -> Path:
        return self.root / "HOI4D_annotations" / self._rel(meta)

    def _hand_dir(self, meta: SequenceMeta, side: str) -> Path:
        return self.root / "Hand_pose" / f"handpose_{side}_hand" / self._rel(meta)

    # ---- discovery ---------------------------------------------------------------------
    def status(self) -> dict:
        st = super().status()
        st.update(
            rgb_extracted=self.release_root.exists(),
            annotations=(self.root / "HOI4D_annotations").exists(),
            hand_pose=(self.root / "Hand_pose").exists(),
        )
        if not self.release_root.exists() and (self.root / "HOI4D_release.zip").exists():
            st["note"] = "Unzip HOI4D_release.zip to index videos."
        return st

    def _videos(self) -> Iterator[Path]:
        for dirpath, dirnames, filenames in os.walk(self.release_root):
            dirnames.sort()
            if "image.mp4" in filenames and dirpath.endswith("align_rgb"):
                yield Path(dirpath) / "image.mp4"

    def _meta_for(self, video: Path) -> SequenceMeta | None:
        rel = str(video.parent.parent.relative_to(self.release_root))
        parts = rel.split("/")
        if len(parts) != 7:
            return None
        try:
            vi = probe(video)
        except Exception:
            return None
        device, human, cat, inst, scene, layout, task = parts
        ann = self.root / "HOI4D_annotations" / rel
        hand = [self.root / "Hand_pose" / f"handpose_{s}_hand" / rel for s in ("left", "right")]
        K_path = self.root / "camera_params" / device / "intrin.npy"
        cname = CATEGORIES.get(cat, cat)
        return SequenceMeta(
            dataset=self.name, sequence_id=rel.replace("/", "-"),
            video_path=str(video.relative_to(self.root)),
            fps=vi.fps or HOI4D_FPS, num_frames=vi.num_frames, width=vi.width, height=vi.height,
            duration_s=vi.duration_s,
            intrinsics=np.load(K_path).tolist() if K_path.exists() else None,
            has_hand_pose=any(h.is_dir() for h in hand),
            has_object_pose=(ann / "objpose").is_dir(),
            has_camera_pose=(ann / "3Dseg" / "output.log").exists(),
            has_actions=(ann / "action" / "color.json").exists(),
            tags={"rel": rel, "device": device, "human": human, "category": cat,
                  "category_name": cname, "rigid": cname in RIGID, "instance": inst,
                  "scene": scene, "layout": layout, "task": task,
                  "hands": [s for s, h in zip(("left", "right"), hand) if h.is_dir()]},
        )

    def scan(self) -> Iterator[SequenceMeta]:
        with ThreadPoolExecutor(16) as pool:  # I/O-bound: probing and stat on a spinning disk
            for meta in pool.map(self._meta_for, self._videos()):
                if meta is not None:
                    yield meta

    # ---- annotations -------------------------------------------------------------------
    def actions(self, meta: SequenceMeta) -> list[ActionSegment]:
        return _load_actions(str(self._ann(meta) / "action" / "color.json"))

    def camera_poses(self, meta: SequenceMeta) -> dict[int, np.ndarray]:
        return _load_cam_log(str(self._ann(meta) / "3Dseg" / "output.log"))

    def _mano_layer(self):
        if self._mano is None and self._want_mano is not False:
            try:
                from egograph.hands.mano import ManoModel
                self._mano = ManoModel()
            except (ImportError, FileNotFoundError):
                if self._want_mano:
                    raise
                self._want_mano = False
        return self._mano

    def load_annotations(self, meta: SequenceMeta, frame_indices) -> Annotations:
        fi = np.asarray(frame_indices, dtype=np.int64)
        ann = Annotations.empty(fi, meta.fps)
        T = len(fi)

        # Camera trajectory: world = camera of the first reconstructed frame.
        poses = self.camera_poses(meta)
        if poses:
            ann.T_world_cam = np.stack([poses.get(int(i), np.full((4, 4), np.nan, np.float32)) for i in fi])

        # Hands
        mano = self._mano_layer()
        for side in ("left", "right"):
            d = self._hand_dir(meta, side)
            if not d.is_dir():
                continue
            h = ann.hands[side]
            for t, i in enumerate(fi):
                p = _frame_file(d, int(i), "pickle")
                if p is None:
                    continue
                with p.open("rb") as f:
                    rec = pickle.load(f)
                h.mano_pose[t] = np.asarray(rec["poseCoeff"], np.float32).reshape(48)
                h.mano_betas[t] = np.asarray(rec["beta"], np.float32).reshape(10)
                h.mano_trans[t] = np.asarray(rec["trans"], np.float32).reshape(3)
                h.joints_2d[t] = np.asarray(rec["kps2D"], np.float32).reshape(21, 2)
                h.valid[t] = True
            if mano is not None and h.valid.any():
                v = h.valid
                h.joints_3d[v] = mano.joints(side, h.mano_pose[v], h.mano_betas[v], h.mano_trans[v])

        # Objects. Articulated categories carry one dataList entry per part.
        tracks: dict[str, ObjectTrack] = {}
        objdir = self._ann(meta) / "objpose"
        for t, i in enumerate(fi):
            p = _frame_file(objdir, int(i), "json")
            if p is None:
                continue
            rec = json.loads(p.read_text())
            entries = rec.get("dataList") or rec.get("objects") or []
            for k, e in enumerate(entries):
                oid = str(e.get("id", k))
                if oid not in tracks:
                    tracks[oid] = ObjectTrack.empty(oid, str(e.get("label", meta.tags.get("category_name"))), T)
                o = tracks[oid]
                c, r, dm = e["center"], e["rotation"], e["dimensions"]
                R = Rotation.from_euler("XYZ", [r["x"], r["y"], r["z"]]).as_matrix()
                o.T_cam_obj[t] = make_T(R, np.array([c["x"], c["y"], c["z"]]))
                o.dimensions = np.array([dm["length"], dm["width"], dm["height"]], np.float32)
                box = (e.get("2dBox") or {}).get("camera1")
                if box:
                    (x0, y0), (x1, y1) = box
                    o.bbox_2d[t] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
                o.valid[t] = True
        ann.objects = list(tracks.values())
        ann.actions = overlapping(self.actions(meta), ann.timestamps)
        return ann


def _frame_file(d: Path, i: int, ext: str) -> Path | None:
    """HOI4D names per-frame files either ``12.json`` or ``00012.json``, depending on the sequence."""
    for name in (f"{i}.{ext}", f"{i:05d}.{ext}"):
        p = d / name
        if p.exists():
            return p
    return None


@lru_cache(maxsize=256)
def _load_actions(path: str) -> list[ActionSegment]:
    p = Path(path)
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    events = d.get("events") or d.get("markResult", {}).get("marks", [])
    out = []
    for e in events:
        label = e.get("event") or e.get("label") or "?"
        out.append(ActionSegment(float(e["startTime"]), float(e["endTime"]), label, verb=label.lower()))
    return sorted(out, key=lambda a: a.start_s)


@lru_cache(maxsize=256)
def _load_cam_log(path: str) -> dict[int, np.ndarray]:
    p = Path(path)
    return parse_open3d_log(p) if p.exists() else {}
