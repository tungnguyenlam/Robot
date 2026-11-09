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
  and HOI4D). ``egograph.hands.mano.ManoModel`` implements this convention exactly.
* A missing value is stored as NaN, with ``valid[t] == False``. Arrays are never
  ragged in time, so a clip with T frames has T rows in every per-frame array.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "egoformat-v1"
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SequenceMeta":
        return cls(**d)

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

    @classmethod
    def empty(cls, side: str, T: int) -> "HandTrack":
        return cls(side, np.zeros(T, bool), _nan((T, 21, 2)), _nan((T, 21, 3)),
                   _nan((T, 48)), _nan((T, 10)), _nan((T, 3)))


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
class Annotations:
    """Everything known about frames ``frame_indices`` of a sequence."""

    frame_indices: np.ndarray                    # (T,) int64 source frame indices
    timestamps: np.ndarray                       # (T,) float64 seconds
    hands: dict[str, HandTrack]                  # keys are HAND_SIDES
    objects: list[ObjectTrack] = field(default_factory=list)
    T_world_cam: np.ndarray | None = None        # (T, 4, 4) camera pose, or None
    actions: list[ActionSegment] = field(default_factory=list)  # segments overlapping the window

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
            for k in ("valid", "joints_2d", "joints_3d", "mano_pose", "mano_betas", "mano_trans"):
                arrays[f"hand_{side}_{k}"] = getattr(h, k)
        objects = []
        for i, o in enumerate(self.objects):
            for k in ("valid", "T_cam_obj", "bbox_2d", "dimensions"):
                arrays[f"obj{i}_{k}"] = getattr(o, k)
            objects.append({"object_id": o.object_id, "label": o.label, "mesh_path": o.mesh_path})
        sidecar = {"objects": objects, "actions": [asdict(a) for a in self.actions]}
        return arrays, sidecar

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], sidecar: dict[str, Any]) -> "Annotations":
        hands = {
            side: HandTrack(side, *(np.asarray(arrays[f"hand_{side}_{k}"]) for k in
                                    ("valid", "joints_2d", "joints_3d", "mano_pose", "mano_betas", "mano_trans")))
            for side in HAND_SIDES
        }
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
        )


@dataclass
class Clip:
    """A decoded window of video plus its annotations: the unit the data loader yields."""

    meta: SequenceMeta
    frames: np.ndarray | None    # (T, H, W, 3) uint8 RGB, or None when only annotations were requested
    K: np.ndarray | None         # (3, 3) intrinsics after any resize or crop
    annotations: Annotations

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
        meta = {
            "schema": SCHEMA_VERSION,
            "source": self.meta.to_dict(),
            "num_frames": self.annotations.num_frames,
            "frame_size": [w, h] if w else None,
            "frame_format": frame_format if self.frames is not None else None,
            "K": None if self.K is None else np.asarray(self.K).tolist(),
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
        frames = None
        if load_frames and meta.get("frame_format"):
            files = sorted((d / "frames").glob(f"*.{meta['frame_format']}"))
            frames = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
        K = None if meta.get("K") is None else np.asarray(meta["K"], dtype=np.float32)
        return cls(SequenceMeta.from_dict(meta["source"]), frames, K, ann)
