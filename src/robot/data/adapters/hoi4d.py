"""HOI4D adapter, the 3D ground-truth source (MANO hands, 6-DoF objects, camera trajectory).

Layout under ``$EGO_DATA_ROOT/HOI4D`` (as downloaded and extracted on amdserver)::

    HOI4D_release/<seq>/align_rgb/image.mp4          # 1920x1080, 15 fps, ~300 frames
    HOI4D_annotations/<seq>/objpose/<frame>.json     # per-frame 6-DoF object box(es)
    HOI4D_annotations/<seq>/action/color.json        # action events (seconds)
    HOI4D_annotations/<seq>/3Dseg/output.log         # camera trajectory (Open3D .log)
    Hand_pose/handpose_{left,right}_hand/<seq>/<frame>.pickle   # MANO poseCoeff/beta/trans/kps2D
    camera_params/<ZY...>/intrin.npy                 # 3x3 K per capture device
    HOI4D_CAD_Model_for_release/rigid/<Category>/<NNN>.obj            # one mesh per instance
    HOI4D_CAD_Model_for_release/articulated/<Category>/<NNN>/objs/new-<k>-align.obj  # one per part
    HOI4D_depth_video/<seq>/align_depth/depth_video.avi   # FFV1 gray16le, mm, registered to RGB
    HOI4D_annotations/<seq>/2Dseg/{mask,shift_mask}/<frame:05d>.png   # 2D instance masks
    HOI4D_annotations/<seq>/3Dseg/label.pcd           # labelled static point cloud (world frame)

``<seq>`` is ``ZY2021080000x/H?/C?/N?/S?/s?/T?``: device / human / category / object
instance / scene / layout / task. The sequence id replaces '/' with '-'.

Conventions, checked against the data with tests/test_hoi4d_conventions.py (2026-11-09):
* objpose ``rotation`` holds intrinsic XYZ Euler angles in radians, ``center`` is in the
  camera frame in meters, and ``dimensions`` (length, width, height) are the extents along
  the object x, y, and z axes. The projected 3D box matches the annotated 2D box with mean
  IoU 0.945 over 140 samples; the next-best convention scores 0.71.
* CAD meshes are in the object frame of objpose: a rigid mesh is centred at the origin and its
  extents along x, y, z match ``dimensions`` (l, w, h) to a few mm. Articulated objects have one
  ``-align`` mesh per part in that part's box frame; ``result.json`` names the parts, and
  ``_match_parts`` pairs them with the annotated part labels (see docs/MODALITIES.md).
* hand ``poseCoeff`` is 48-D axis-angle (no PCA, flat_hand_mean=True) and ``trans`` is in
  the camera frame. ``kps2D`` is in the 21-joint OpenPose order. Projected MANO joints are
  0.7 px (right) and 0.85 px (left) from kps2D (median). With flat_hand_mean=False the error
  is 32 px; with the smplx left-shapedirs fix it is 29 px.
* depth is uint16 millimetres, 1920x1080 at 15 fps, pixel-aligned with the RGB video: at
  the projected object centre the depth is 0.909 m for an object whose centre is 0.921 m
  away (the visible surface is in front of the centre). 0 means no reading.
* 2D masks are PNGs, either palette ("mask") or RGB ("shift_mask", ~2/3 of sequences), both
  using the PASCAL-VOC colour map. We return the VOC index as the instance id. The ids are
  per sequence, not global classes: over 1,000 sampled frames the right hand is id 2
  (0,128,0), the left hand id 3 (128,128,0) in two-hand sequences, and the main object id 1
  (128,0,0), but articulated parts also use 3, 4 (0,0,128), and 5 (128,0,128).
* ``3Dseg/label.pcd`` is ASCII with fields x y z rgb label ID, in the same world frame as
  ``3Dseg/output.log`` (the first reconstructed camera).
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

from robot.data.adapters.base import DatasetAdapter, overlapping, register
from robot.data.geometry import make_T
from robot.data.schema import (ActionSegment, Annotations, ObjectMesh, ObjectTrack, ScenePoints,
                                  SequenceMeta)
from robot.data.video import probe

CATEGORIES = {
    "C1": "ToyCar", "C2": "Mug", "C3": "Laptop", "C4": "StorageFurniture", "C5": "Bottle",
    "C6": "Safe", "C7": "Bowl", "C8": "Bucket", "C9": "Scissors", "C11": "Pliers",
    "C12": "Kettle", "C13": "Knife", "C14": "TrashCan", "C17": "Lamp", "C18": "Stapler",
    "C20": "Chair",
}
RIGID = {"ToyCar", "Mug", "Bottle", "Bowl", "Kettle", "Knife"}

HOI4D_FPS = 15.0
CAD_DIR = "HOI4D_CAD_Model_for_release"

# Annotated part label (lower case) -> words that the matching CAD part name contains.
# Labels not listed here (scissors/scissors2, the lamp joints) are matched by box extents alone.
PART_WORDS = {
    "laptopdisplay": ("screen",), "laptopkeyboard": ("base",),
    "lockerbody": ("panel", "body", "frame", "base"), "lockerdrawer": ("drawer",),
    "lockersldingdoor": ("door",), "safebox": ("frame", "body"), "safedoor": ("door",),
    "bucket": ("body",), "buckethandle": ("handle",), "dustbinbase": ("body", "base"),
    "dustbincover": ("lid", "cover"), "staplerbase": ("body", "base"), "staplercover": ("lid", "cover"),
    "pliersleft": ("left",), "pliersright": ("right",), "lampbase": ("base",),
}


def voc_palette(n: int = 256) -> np.ndarray:
    """The PASCAL-VOC colour map: (n, 3) uint8, index -> RGB."""
    pal = np.zeros((n, 3), np.uint8)
    for i in range(n):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        pal[i] = (r, g, b)
    return pal


_VOC_LUT = {tuple(int(x) for x in c): i for i, c in enumerate(voc_palette())}


def mask_ids(path: Path) -> np.ndarray:
    """Read a HOI4D 2D mask as (H, W) uint8 VOC indices, whether it is a palette or RGB PNG."""
    from PIL import Image

    im = Image.open(path)
    if im.mode == "P":
        return np.asarray(im, dtype=np.uint8)
    rgb = np.asarray(im.convert("RGB"))
    key = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
    out = np.zeros(key.shape, np.uint8)
    for k in np.unique(key):
        idx = _VOC_LUT.get(((k >> 16) & 255, (k >> 8) & 255, k & 255), 255)  # 255: colour outside the map
        out[key == k] = idx
    return out


def read_pcd(path: Path) -> dict[str, np.ndarray]:
    """Minimal PCD reader for HOI4D's ASCII clouds. Returns one array per field."""
    with path.open("rb") as f:
        fields: list[str] = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            if line.startswith("DATA"):
                if line.split()[1] != "ascii":
                    raise ValueError(f"{path}: only ASCII PCD is supported")
                break
        data = np.loadtxt(f, dtype=np.float64, ndmin=2)
    return {name: data[:, i] for i, name in enumerate(fields)}


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

    def _depth_path(self, rel: str) -> Path:
        return self.root / "HOI4D_depth_video" / rel / "align_depth" / "depth_video.avi"

    def _seg_dir(self, rel: str) -> Path | None:
        for kind in ("mask", "shift_mask"):
            d = self.root / "HOI4D_annotations" / rel / "2Dseg" / kind
            if d.is_dir():
                return d
        return None

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
        has_hand = any(h.is_dir() for h in hand)
        flags = {
            "rgb": True, "intrinsics": K_path.exists(), "hand_pose": has_hand,
            "object_pose": (ann / "objpose").is_dir(), "object_box": (ann / "objpose").is_dir(),
            "camera_pose": (ann / "3Dseg" / "output.log").exists(),
            "actions": (ann / "action" / "color.json").exists(),
            "depth": self._depth_path(rel).exists(), "seg": self._seg_dir(rel) is not None,
            "scene": (ann / "3Dseg" / "label.pcd").exists(),
            "mesh": self._cad_path(cname, inst) is not None,
        }
        flags["contact"] = flags["hand_pose"] and flags["object_pose"]
        return SequenceMeta(
            dataset=self.name, sequence_id=rel.replace("/", "-"),
            video_path=str(video.relative_to(self.root)),
            fps=vi.fps or HOI4D_FPS, num_frames=vi.num_frames, width=vi.width, height=vi.height,
            duration_s=vi.duration_s,
            intrinsics=np.load(K_path).tolist() if K_path.exists() else None,
            has_hand_pose=has_hand,
            has_object_pose=(ann / "objpose").is_dir(),
            has_camera_pose=(ann / "3Dseg" / "output.log").exists(),
            has_actions=(ann / "action" / "color.json").exists(),
            tags={"rel": rel, "device": device, "human": human, "category": cat,
                  "category_name": cname, "rigid": cname in RIGID, "instance": inst,
                  "scene": scene, "layout": layout, "task": task,
                  "hands": [s for s, h in zip(("left", "right"), hand) if h.is_dir()]},
            modalities=[k for k, v in flags.items() if v],
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

    # ---- meshes ---------------------------------------------------------------------------
    def _cad_path(self, cname: str, inst: str) -> Path | None:
        """The rigid .obj or the articulated instance folder of one object, if downloaded."""
        try:
            n = f"{int(inst.lstrip('N')):03d}"
        except ValueError:
            return None
        for p in (self.root / CAD_DIR / "rigid" / cname / f"{n}.obj",
                  self.root / CAD_DIR / "articulated" / cname / n):
            if p.exists():
                return p
        return None

    def _assign_meshes(self, meta: SequenceMeta, objects: list[ObjectTrack]) -> None:
        """Set ``mesh_path`` (relative to the dataset root) on each object track."""
        cad = self._cad_path(meta.tags.get("category_name", ""), meta.tags.get("instance", ""))
        if cad is None or not objects:
            return
        if cad.is_file():  # rigid: one object per sequence
            objects[0].mesh_path = str(cad.relative_to(self.root))
            return
        key = tuple((o.label, tuple(float(x) for x in np.round(o.dimensions, 3))) for o in objects)
        for i, part in _match_parts(str(cad), key):
            if part is not None:
                objects[i].mesh_path = str((cad / "objs" / f"{part}-align.obj").relative_to(self.root))

    def load_meshes(self, meta: SequenceMeta, ann: Annotations) -> dict[str, ObjectMesh] | None:
        from robot.data.mesh import load_mesh

        out = {o.object_id: load_mesh(self.root / o.mesh_path, self.name, o.mesh_path)
               for o in ann.objects if o.mesh_path}
        return out or None

    # ---- dense modalities ----------------------------------------------------------------
    def load_depth(self, meta: SequenceMeta, frame_indices) -> np.ndarray | None:
        p = self._depth_path(self._rel(meta))
        if not p.exists():
            return None
        from robot.data.video import VideoReader

        with VideoReader(p, pixel_format="gray16le") as vr:
            return vr.read(np.asarray(frame_indices, np.int64)).astype(np.float32) / 1000.0

    def load_seg(self, meta: SequenceMeta, frame_indices) -> np.ndarray | None:
        d = self._seg_dir(self._rel(meta))
        if d is None:
            return None
        out = np.zeros((len(frame_indices), meta.height, meta.width), np.uint8)
        for t, i in enumerate(frame_indices):
            p = d / f"{int(i):05d}.png"
            if p.exists():
                out[t] = mask_ids(p)
        return out

    def load_scene(self, meta: SequenceMeta) -> ScenePoints | None:
        return _load_scene(str(self._ann(meta) / "3Dseg" / "label.pcd"))

    def _mano_layer(self):
        if self._mano is None and self._want_mano is not False:
            try:
                from robot.hands.mano import ManoModel
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
        self._assign_meshes(meta, ann.objects)
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
        # 2,955 files use startTime/endTime; 17 use the "markResult" layout with hdTimeStart/hdTimeEnd.
        start, end = e.get("startTime", e.get("hdTimeStart")), e.get("endTime", e.get("hdTimeEnd"))
        if start is None or end is None:
            continue
        out.append(ActionSegment(float(start), float(end), label, verb=label.lower()))
    return sorted(out, key=lambda a: a.start_s)


@lru_cache(maxsize=256)
def _load_cam_log(path: str) -> dict[int, np.ndarray]:
    p = Path(path)
    return parse_open3d_log(p) if p.exists() else {}


@lru_cache(maxsize=1024)
def _match_parts(cad_dir: str, objects: tuple) -> list[tuple[int, str | None]]:
    """Pair annotated parts ``objects`` = ((label, dims), ...) with the CAD parts of an articulated
    instance folder. Cost = L1 gap (m) between box dims and the part mesh's extents, plus 1 when
    the label has known part words (PART_WORDS) and the part name has none of them. Solved as an
    assignment, so two annotated parts never share a mesh. Returns (object index, obj stem)."""
    from scipy.optimize import linear_sum_assignment

    from robot.data.mesh import load_mesh

    d = Path(cad_dir)
    root = d.parents[3]  # <HOI4D>/HOI4D_CAD_Model_for_release/articulated/<Category>/<NNN>
    parts: list[tuple[str, str]] = []  # (part name, obj stem)

    def walk(n):
        for obj in n.get("objs", []):
            parts.append((n.get("name", ""), obj))
        for c in n.get("children", []):
            walk(c)
    for n in json.loads((d / "result.json").read_text()):
        walk(n)
    parts = [(name, obj) for name, obj in parts if (d / "objs" / f"{obj}-align.obj").exists()]
    if not parts:
        return [(i, None) for i in range(len(objects))]
    paths = [d / "objs" / f"{obj}-align.obj" for _, obj in parts]
    ext = [np.ptp(load_mesh(p, "hoi4d", str(p.relative_to(root))).vertices, 0) for p in paths]
    cost = np.zeros((len(objects), len(parts)))
    for i, (label, dims) in enumerate(objects):
        words = PART_WORDS.get(str(label).lower())
        for j, (name, _) in enumerate(parts):
            if np.isfinite(dims).all():
                cost[i, j] = np.abs(np.asarray(dims) - ext[j]).sum()
            if words and not any(w in name.lower() for w in words):
                cost[i, j] += 1.0
    rows, cols = linear_sum_assignment(cost)
    got = dict(zip(rows.tolist(), cols.tolist()))
    return [(i, parts[got[i]][1] if i in got else None) for i in range(len(objects))]


@lru_cache(maxsize=8)
def _load_scene(path: str) -> ScenePoints | None:
    p = Path(path)
    if not p.exists():
        return None
    f = read_pcd(p)
    rgb = f["rgb"].astype(np.uint32)  # packed 0x00RRGGBB
    return ScenePoints(
        xyz=np.stack([f["x"], f["y"], f["z"]], 1).astype(np.float32),
        rgb=np.stack([(rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255], 1).astype(np.uint8),
        label=f.get("label", np.full(len(rgb), -1)).astype(np.int32),
        instance=f["ID"].astype(np.int32) if "ID" in f else None,
    )
