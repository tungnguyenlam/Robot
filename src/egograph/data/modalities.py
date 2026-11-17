"""Registry of every data modality the loader can produce.

A modality name appears in three places:

* ``SequenceMeta.modalities``: what one sequence has, filled by its adapter at index time.
* ``EgoLoader(modalities=...)`` and the ``data.modalities`` config key: what to load.
* the tensor keys of ``clip_to_tensors``: the table below lists them.

"Labels" (hands, objects, camera, actions) come from ``adapter.load_annotations`` and are
cheap, so they are always loaded. The other modalities cost I/O and are loaded only on request.
See docs/MODALITIES.md for where each modality comes from and which baseline uses it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Modality:
    name: str
    kind: str          # "dense" (per-pixel per-frame), "frame" (per-frame vector), "signal", "sequence", "label"
    on_request: bool   # loaded only when asked for (True), or always with the annotations (False)
    tensors: str       # keys added by clip_to_tensors
    description: str


MODALITIES: dict[str, Modality] = {m.name: m for m in [
    Modality("rgb", "dense", True, "frames (T,3,H,W)",
             "RGB video frames"),
    Modality("depth", "dense", True, "depth (T,1,H,W) metres, depth_valid",
             "metric depth registered to the RGB pixels"),
    Modality("seg", "dense", True, "seg (T,H,W) uint8 ids, seg_valid",
             "per-pixel instance ids (hands and object parts)"),
    Modality("audio", "signal", True, "audio (C,S), audio_valid",
             "the video's audio track, resampled and cut to the clip span"),
    Modality("imu", "signal", True, "imu (T,6) [accel, gyro], imu_valid (T,), gravity_cam (3,)",
             "accelerometer and gyroscope; interpolated to frame times in tensors"),
    Modality("gaze", "frame", True, "gaze_2d (T,2), gaze_valid (T,)",
             "2D eye-gaze point in pixels"),
    Modality("scene", "sequence", True, "scene_points (N,6) xyz+rgb, scene_valid",
             "static point cloud of the scene in the world frame"),
    Modality("mesh", "sequence", True, "object_points (O,P,6) xyz+normal in the object frame, object_points_valid (O,)",
             "object (part) meshes in the frame of T_cam_obj: simplified mesh + surface samples"),
    Modality("contact", "frame", True, "contact (2,O,T) bool, contact_dist (2,O,T) metres",
             "hand-object proximity derived from 3D joints and object boxes"),
    Modality("hand_pose", "label", False, "hand_valid, hand_joints_2d/3d, hand_mano",
             "MANO parameters, 3D joints, and 2D keypoints per hand"),
    Modality("hand_box", "label", False, "hand_bbox_2d (2,T,4)",
             "2D hand boxes (sparse key frames)"),
    Modality("object_pose", "label", False, "obj_valid, obj_T_cam_obj, obj_dims",
             "6-DoF object (or part) poses with box dimensions"),
    Modality("object_box", "label", False, "obj_bbox_2d (O,T,4)",
             "2D object boxes"),
    Modality("camera_pose", "label", False, "cam_valid, T_world_cam",
             "camera trajectory (camera-to-world)"),
    Modality("intrinsics", "label", False, "K, K_valid",
             "pinhole intrinsics at the native resolution"),
    Modality("actions", "label", False, "action",
             "action segments, narrations, and verb/noun labels"),
]}

ON_REQUEST = tuple(n for n, m in MODALITIES.items() if m.on_request)


def check(names: list[str] | tuple[str, ...] | set[str]) -> set[str]:
    """Validate a list of requested modality names."""
    bad = set(names) - set(MODALITIES)
    if bad:
        raise ValueError(f"unknown modalities {sorted(bad)}; known: {sorted(MODALITIES)}")
    return set(names)
