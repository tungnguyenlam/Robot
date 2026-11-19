"""Standard egocentric hand-object format ("egoformat v1").

Every dataset adapter parses its raw files into these containers, so everything
downstream (pose extractors, graph builders, diffusion models, the web viewer)
sees a single representation.

Conventions
-----------
* Units: meters, seconds, radians. Image coordinates are pixels, origin at the
  top-left corner, x to the right, y down.
* Camera frame: OpenCV (x right, y down, z forward).
* ``T_a_b`` is a 4x4 homogeneous transform that maps points from frame ``b`` to
  frame ``a`` (``p_a = T_a_b @ p_b``). For example ``T_world_cam`` is the camera
  pose in world coordinates and ``T_cam_obj`` is the object pose in the camera frame.
* Hand joints use the 21-joint OpenPose order: wrist, then thumb, index, middle,
  ring, and pinky, each listed base to tip (see ``HAND_JOINT_NAMES``).
* MANO parameters: ``pose`` is 48-D axis-angle (3 global orientation + 45 finger),
  without PCA, and absolute (``flat_hand_mean=True``: a zero pose is a flat hand).
  ``trans`` is in the same camera frame as the joints. A source that uses the
  mean-relative convention must add ``hands_mean`` when converting. Left-hand betas
  refer to the original MANO_LEFT.pkl, without the smplx shapedirs fix (as in manopth
  and HOI4D). ``robot.hands.mano.ManoModel`` implements this convention exactly.
* A missing value is stored as NaN, with ``valid[t] == False``. Arrays are never
  ragged in time, so a clip with T frames has T rows in every per-frame array. The one
  exception is IMU, which keeps its native sample rate (``ImuTrack``) because resampling
  it to the video rate would throw information away.
* Dense per-frame modalities (depth, segmentation) live on ``Clip`` next to ``frames``
  and share its (T, H, W) grid, so the same transforms keep them aligned.

The full list of modalities and which dataset provides each one is in
``robot.data.modalities`` and docs/MODALITIES.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "egoformat-v1.1"  # v1.1 adds optional modalities; v1 files still load
HAND_SIDES = ("left", "right")
NUM_HAND_JOINTS = 21
HAND_JOINT_NAMES = (
    "wrist",
    "thumb1", "thumb2", "thumb3", "thumb4",
    "index1", "index2", "index3", "index4",
    "middle1", "middle2", "middle3", "middle4",
    "ring1", "ring2", "ring3", "ring4",
    "pinky1", "pinky2", "pinky3", "pinky4",
)
# Parent of each joint in HAND_JOINT_NAMES (-1 for the root). Used to draw skeletons
# and later as the kinematic edge set of the hand graph.
HAND_PARENTS = (-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19)
HAND_FIELDS = ("valid", "joints_2d", "joints_3d", "mano_pose", "mano_betas", "mano_trans", "bbox_2d")


def _nan(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, np.nan, dtype=np.float32)


# --------------------------------------------------------------------------- metadata
@dataclass
class SequenceMeta:
    """Static description of one raw egocentric video. Indexes store one per line as JSON."""

    dataset: str
    sequence_id: str            # unique within the dataset and URL-safe (no '/')
    video_path: str             # relative to the dataset root
    fps: float
    num_frames: int
    width: int
    height: int
    duration_s: float
    intrinsics: list[list[float]] | None = None  # 3x3 K at native resolution, if known
    has_hand_pose: bool = False
    has_object_pose: bool = False
    has_camera_pose: bool = False
    has_actions: bool = False
    tags: dict[str, Any] = field(default_factory=dict)  # dataset-specific (participant, category, ...)
    # Every modality available for this sequence, by the names in robot.data.modalities
    # (for example ["rgb", "audio", "imu", "depth"]). Filled by the adapter at index time.
    modalities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SequenceMeta":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    def has(self, modality: str) -> bool:
        return modality in self.modalities

    @property
    def K(self) -> np.ndarray | None:
        return None if self.intrinsics is None else np.asarray(self.intrinsics, dtype=np.float32)


@dataclass
class ActionSegment:
    start_s: float
    end_s: float
    label: str                  # free-form narration or event name
    verb: str | None = None
    noun: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- tracks
@dataclass
class HandTrack:
    """Per-frame annotation of one hand. Any field can be all-NaN when the dataset lacks it."""

    side: str
    valid: np.ndarray           # (T,) bool: the hand is annotated in this frame
    joints_2d: np.ndarray       # (T, 21, 2) pixels
    joints_3d: np.ndarray       # (T, 21, 3) camera frame, meters
    mano_pose: np.ndarray       # (T, 48)
    mano_betas: np.ndarray      # (T, 10)
    mano_trans: np.ndarray      # (T, 3)
    bbox_2d: np.ndarray | None = None  # (T, 4) xyxy pixels: a hand box (Ego4D FHO); NaN when absent

    def __post_init__(self) -> None:
        if self.bbox_2d is None:
            self.bbox_2d = _nan((len(self.valid), 4))

    @classmethod
    def empty(cls, side: str, T: int) -> "HandTrack":
        return cls(side, np.zeros(T, bool), _nan((T, 21, 2)), _nan((T, 21, 3)),
                   _nan((T, 48)), _nan((T, 10)), _nan((T, 3)), _nan((T, 4)))


@dataclass
class ObjectTrack:
    """Per-frame 6-DoF pose and box of one object (or one articulated part)."""

    object_id: str
    label: str
    valid: np.ndarray           # (T,) bool
    T_cam_obj: np.ndarray       # (T, 4, 4) object-to-camera transform
    bbox_2d: np.ndarray         # (T, 4) xyxy pixels
    dimensions: np.ndarray      # (3,) box extents along the object x, y, z axes, meters
    mesh_path: str | None = None

    @classmethod
    def empty(cls, object_id: str, label: str, T: int) -> "ObjectTrack":
        return cls(object_id, label, np.zeros(T, bool), _nan((T, 4, 4)), _nan((T, 4)), _nan((3,)))


@dataclass
class ImuTrack:
    """Inertial samples at the sensor's native rate, restricted to the clip's time span.

    ``frame`` names the axes: "camera" means the OpenCV camera frame (x right, y down,
    z forward), and "device" means the IMU's own body frame, whose relation to the camera
    is unknown. ``accel`` is specific force (gravity included: at rest it points up).
    """

    t: np.ndarray               # (N,) float64 seconds on the video timeline
    accel: np.ndarray           # (N, 3) m/s^2
    gyro: np.ndarray            # (N, 3) rad/s
    frame: str = "device"

    def at(self, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linearly interpolate to ``timestamps``. Returns ((T, 6) [accel, gyro], (T,) valid).
        A timestamp is valid when samples exist within 50 ms on both sides of it."""
        ts = np.asarray(timestamps, np.float64)
        out = np.full((len(ts), 6), np.nan, np.float32)
        if len(self.t) < 2:
            return out, np.zeros(len(ts), bool)
        both = np.concatenate([self.accel, self.gyro], axis=1)
        for k in range(6):
            out[:, k] = np.interp(ts, self.t, both[:, k])
        j = np.clip(np.searchsorted(self.t, ts), 1, len(self.t) - 1)
        valid = (ts >= self.t[0]) & (ts <= self.t[-1]) & (self.t[j] - self.t[j - 1] < 0.05)
        out[~valid] = np.nan
        return out, valid

    def window(self, t0: float, t1: float) -> "ImuTrack":
        k = (self.t >= t0) & (self.t <= t1)
        return ImuTrack(self.t[k], self.accel[k], self.gyro[k], self.frame)


@dataclass
class ScenePoints:
    """A static, sequence-level point cloud in the world frame of ``T_world_cam``
    (HOI4D 3Dseg reconstruction). ``label`` holds per-point semantic ids, -1 when unknown."""

    xyz: np.ndarray             # (N, 3) float32 metres, world frame
    rgb: np.ndarray             # (N, 3) uint8
    label: np.ndarray           # (N,) int32
    instance: np.ndarray | None = None  # (N,) int32 instance ids, if the source has them


@dataclass
class ObjectMesh:
    """Triangle mesh of one object (or one articulated part) in its own frame, the frame of
    ``ObjectTrack.T_cam_obj``: ``T_cam_obj @ [v, 1]`` puts a vertex in the camera frame.

    ``vertices``/``faces`` are simplified for drawing (see robot.data.mesh); ``points`` and
    ``normals`` are an area-weighted sample of the full-resolution surface, in random order, so
    the first n rows are themselves a uniform sample."""

    vertices: np.ndarray        # (V, 3) float32 metres
    faces: np.ndarray           # (F, 3) int32
    points: np.ndarray          # (P, 3) float32 surface samples
    normals: np.ndarray         # (P, 3) float32 unit outward normals at ``points``
    source: str = ""            # path of the source mesh, relative to the dataset root


# PolygonMasks.contact codes (values >= 0 are entity ids).
CONTACT_NA = -1            # not a hand, or no contact label on this frame (interpolated frames)
CONTACT_NONE = -2          # hand not in contact
CONTACT_UNSEGMENTED = -3   # hand touches something that has no mask ("none-of-the-above")
CONTACT_UNSURE = -4        # annotators disagreed ("inconclusive")


@dataclass
class PolygonMasks:
    """Instance masks stored as polygons on a subset of a sequence's frames (EPIC VISOR).

    Frames are sorted source-frame indices. Masks of frame ``j`` are
    ``frame_start[j]:frame_start[j+1]``; polygons of mask ``m`` are
    ``mask_start[m]:mask_start[m+1]``; points of polygon ``p`` are
    ``points[poly_start[p]:poly_start[p+1]]``. ``entity`` indexes ``names``, which are stable
    within a sequence (mask ids in the source are not). ``contact`` is set on hand masks of
    manually annotated frames: the entity the hand touches, or one of the ``CONTACT_*`` codes."""

    names: list[str]            # entity names; index = entity id
    frames: np.ndarray          # (F,) int64 source frame indices
    manual: np.ndarray          # (F,) bool, True = manual annotation, False = interpolated
    frame_start: np.ndarray     # (F+1,) int64
    entity: np.ndarray          # (M,) int32
    contact: np.ndarray         # (M,) int32
    mask_start: np.ndarray      # (M+1,) int64
    poly_start: np.ndarray      # (P+1,) int64
    points: np.ndarray          # (N, 2) int16 pixels at the native video resolution

    def window(self, start: int, end: int) -> list[dict]:
        """Frames in ``[start, end)`` as ``[{"frame", "manual", "masks": [{"entity", "contact",
        "polygons": [(K,2) int16, ...]}]}]``."""
        out = []
        for j in range(int(np.searchsorted(self.frames, start)), int(np.searchsorted(self.frames, end))):
            masks = []
            for m in range(self.frame_start[j], self.frame_start[j + 1]):
                polys = [self.points[self.poly_start[p]:self.poly_start[p + 1]]
                         for p in range(self.mask_start[m], self.mask_start[m + 1])]
                masks.append({"entity": int(self.entity[m]), "contact": int(self.contact[m]), "polygons": polys})
            out.append({"frame": int(self.frames[j]), "manual": bool(self.manual[j]), "masks": masks})
        return out


@dataclass
class Annotations:
    """Everything known about frames ``frame_indices`` of a sequence."""

    frame_indices: np.ndarray                    # (T,) int64 source frame indices
    timestamps: np.ndarray                       # (T,) float64 seconds
    hands: dict[str, HandTrack]                  # keys are HAND_SIDES
    objects: list[ObjectTrack] = field(default_factory=list)
    T_world_cam: np.ndarray | None = None        # (T, 4, 4) camera pose, or None
    actions: list[ActionSegment] = field(default_factory=list)  # segments overlapping the window
    gaze_2d: np.ndarray | None = None            # (T, 2) pixels where the wearer looks; NaN if unknown
    imu: ImuTrack | None = None                  # native-rate samples covering the window

    @property
    def num_frames(self) -> int:
        return int(len(self.frame_indices))

    @classmethod
    def empty(cls, frame_indices: np.ndarray, fps: float) -> "Annotations":
        fi = np.asarray(frame_indices, dtype=np.int64)
        return cls(fi, fi / float(fps), {s: HandTrack.empty(s, len(fi)) for s in HAND_SIDES})

    # ---- flat (de)serialization: npz arrays + a JSON sidecar for strings and lists ----
    def to_arrays(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays: dict[str, np.ndarray] = {
            "frame_indices": self.frame_indices,
            "timestamps": self.timestamps,
        }
        if self.T_world_cam is not None:
            arrays["T_world_cam"] = self.T_world_cam.astype(np.float32)
        for side, h in self.hands.items():
            for k in HAND_FIELDS:
                arrays[f"hand_{side}_{k}"] = getattr(h, k)
        if self.gaze_2d is not None:
            arrays["gaze_2d"] = self.gaze_2d.astype(np.float32)
        if self.imu is not None:
            arrays["imu_t"], arrays["imu_accel"], arrays["imu_gyro"] = self.imu.t, self.imu.accel, self.imu.gyro
        objects = []
        for i, o in enumerate(self.objects):
            for k in ("valid", "T_cam_obj", "bbox_2d", "dimensions"):
                arrays[f"obj{i}_{k}"] = getattr(o, k)
            objects.append({"object_id": o.object_id, "label": o.label, "mesh_path": o.mesh_path})
        sidecar = {"objects": objects, "actions": [asdict(a) for a in self.actions],
                   "imu_frame": self.imu.frame if self.imu is not None else None}
        return arrays, sidecar

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], sidecar: dict[str, Any]) -> "Annotations":
        hands = {  # bbox_2d is absent from v1 files, so HandTrack fills it with NaN
            side: HandTrack(side, *(np.asarray(arrays[f"hand_{side}_{k}"]) if f"hand_{side}_{k}" in arrays else None
                                    for k in HAND_FIELDS))
            for side in HAND_SIDES
        }
        imu = (ImuTrack(np.asarray(arrays["imu_t"]), np.asarray(arrays["imu_accel"]), np.asarray(arrays["imu_gyro"]),
                        sidecar.get("imu_frame") or "device") if "imu_t" in arrays else None)
        objects = [
            ObjectTrack(o["object_id"], o["label"], *(np.asarray(arrays[f"obj{i}_{k}"]) for k in
                                                     ("valid", "T_cam_obj", "bbox_2d", "dimensions")),
                        mesh_path=o.get("mesh_path"))
            for i, o in enumerate(sidecar.get("objects", []))
        ]
        return cls(
            frame_indices=np.asarray(arrays["frame_indices"]),
            timestamps=np.asarray(arrays["timestamps"]),
            hands=hands,
            objects=objects,
            T_world_cam=np.asarray(arrays["T_world_cam"]) if "T_world_cam" in arrays else None,
            actions=[ActionSegment(**a) for a in sidecar.get("actions", [])],
            gaze_2d=np.asarray(arrays["gaze_2d"]) if "gaze_2d" in arrays else None,
            imu=imu,
        )


@dataclass
class Clip:
    """A decoded window of video plus its annotations: the unit the data loader yields.

    Only ``meta`` and ``annotations`` are always set. Every other modality is None unless
    it was requested from the loader and the dataset has it.
    """

    meta: SequenceMeta
    frames: np.ndarray | None    # (T, H, W, 3) uint8 RGB, or None when only annotations were requested
    K: np.ndarray | None         # (3, 3) intrinsics after any resize or crop
    annotations: Annotations
    depth: np.ndarray | None = None      # (T, H, W) float32 metres on the RGB grid; 0 = no reading
    seg: np.ndarray | None = None        # (T, H, W) uint8 instance ids on the RGB grid; 0 = background
    audio: np.ndarray | None = None      # (S, C) float32 in [-1, 1], from the first frame's timestamp on
    audio_sr: int | None = None          # audio sample rate in Hz
    scene: ScenePoints | None = None     # static point cloud of the whole sequence (world frame)
    meshes: dict[str, ObjectMesh] | None = None  # object_id -> mesh in that object's frame

    def save(self, out_dir: str | Path, frame_format: str = "jpg", jpeg_quality: int = 92) -> Path:
        """Write the clip in the on-disk standard format (see docs/DATA_FORMAT.md)."""
        from PIL import Image

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        arrays, sidecar = self.annotations.to_arrays()
        np.savez_compressed(out / "annotations.npz", **arrays)
        h = w = None
        if self.frames is not None:
            fdir = out / "frames"
            fdir.mkdir(exist_ok=True)
            for i, f in enumerate(self.frames):
                img = Image.fromarray(f)
                if frame_format == "jpg":
                    img.save(fdir / f"{i:06d}.jpg", quality=jpeg_quality)
                else:
                    img.save(fdir / f"{i:06d}.png")
            h, w = self.frames.shape[1:3]
        if self.depth is not None:  # lossless 16-bit PNG in millimetres, the HOI4D native format
            ddir = out / "depth"
            ddir.mkdir(exist_ok=True)
            for i, d in enumerate(self.depth):
                Image.fromarray(np.clip(np.nan_to_num(d) * 1000.0 + 0.5, 0, 65535).astype(np.uint16)).save(
                    ddir / f"{i:06d}.png")
        if self.seg is not None:
            sdir = out / "seg"
            sdir.mkdir(exist_ok=True)
            for i, m in enumerate(self.seg):
                Image.fromarray(m.astype(np.uint8)).save(sdir / f"{i:06d}.png")
        extra: dict[str, np.ndarray] = {}
        if self.audio is not None:
            extra["audio"] = self.audio.astype(np.float32)
        if self.scene is not None:
            extra.update(scene_xyz=self.scene.xyz, scene_rgb=self.scene.rgb, scene_label=self.scene.label)
            if self.scene.instance is not None:
                extra["scene_instance"] = self.scene.instance
        if extra:
            np.savez_compressed(out / "extra.npz", **extra)
        if self.meshes:
            mdir = out / "meshes"
            mdir.mkdir(exist_ok=True)
            for oid, m in self.meshes.items():
                np.savez_compressed(mdir / f"{oid}.npz", vertices=m.vertices, faces=m.faces, points=m.points,
                                    normals=m.normals, source=np.array(m.source))
        meta = {
            "schema": SCHEMA_VERSION,
            "source": self.meta.to_dict(),
            "num_frames": self.annotations.num_frames,
            "frame_size": [w, h] if w else None,
            "frame_format": frame_format if self.frames is not None else None,
            "K": None if self.K is None else np.asarray(self.K).tolist(),
            "depth": self.depth is not None,
            "seg": self.seg is not None,
            "audio_sr": self.audio_sr if self.audio is not None else None,
            "meshes": sorted(self.meshes) if self.meshes else None,
            **sidecar,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return out

    @classmethod
    def load(cls, clip_dir: str | Path, load_frames: bool = True) -> "Clip":
        from PIL import Image

        d = Path(clip_dir)
        meta = json.loads((d / "meta.json").read_text())
        with np.load(d / "annotations.npz") as z:
            arrays = {k: z[k] for k in z.files}
        ann = Annotations.from_arrays(arrays, meta)

        def stack(sub: str, ext: str) -> np.ndarray:
            return np.stack([np.asarray(Image.open(f)) for f in sorted((d / sub).glob(f"*.{ext}"))])

        frames = depth = seg = None
        if load_frames and meta.get("frame_format"):
            files = sorted((d / "frames").glob(f"*.{meta['frame_format']}"))
            frames = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
        if load_frames and meta.get("depth"):
            depth = stack("depth", "png").astype(np.float32) / 1000.0
        if load_frames and meta.get("seg"):
            seg = stack("seg", "png").astype(np.uint8)
        audio = scene = None
        if (d / "extra.npz").exists():
            with np.load(d / "extra.npz") as z:
                if "audio" in z.files:
                    audio = z["audio"]
                if "scene_xyz" in z.files:
                    scene = ScenePoints(z["scene_xyz"], z["scene_rgb"], z["scene_label"],
                                        z["scene_instance"] if "scene_instance" in z.files else None)
        meshes = None
        if meta.get("meshes"):
            meshes = {}
            for oid in meta["meshes"]:
                with np.load(d / "meshes" / f"{oid}.npz") as z:
                    meshes[oid] = ObjectMesh(z["vertices"], z["faces"], z["points"], z["normals"], str(z["source"]))
        K = None if meta.get("K") is None else np.asarray(meta["K"], dtype=np.float32)
        return cls(SequenceMeta.from_dict(meta["source"]), frames, K, ann, depth=depth, seg=seg,
                   audio=audio, audio_sr=meta.get("audio_sr"), scene=scene, meshes=meshes)
