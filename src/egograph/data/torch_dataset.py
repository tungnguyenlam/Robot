"""PyTorch ``Dataset`` over standard clips. It needs the ``torch`` extra.

    from egograph.data.torch_dataset import EgoClipDataset
    ds = EgoClipDataset.from_config(load_config("configs/experiment/example_dataloader.yaml")["data"])
    batch = next(iter(DataLoader(ds, batch_size=4, num_workers=4)))
    batch["frames"].shape  # (B, T, 3, H, W)

Every sample has the same keys and shapes, so the default collate works, even when the batch
mixes datasets. Objects are padded to ``max_objects``. A requested modality that a sample's
dataset lacks is returned as zeros with its ``*_valid`` flag False. Missing values become 0,
so always mask with the ``*_valid`` flags. The key list is in docs/DATA_FORMAT.md.
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from egograph.data.contact import DEFAULT_THRESHOLD_M, hand_object_contact
from egograph.data.loader import ClipWindow, EgoLoader, sample_windows
from egograph.data.modalities import check
from egograph.data.schema import HAND_SIDES, Clip
from egograph.data.transforms import build_transforms


def _f32(x) -> torch.Tensor:
    return torch.from_numpy(np.nan_to_num(np.asarray(x, np.float32), nan=0.0, posinf=0.0, neginf=0.0))


def clip_to_tensors(clip: Clip, max_objects: int = 4, modalities: set[str] | None = None,
                    audio_samples: int | None = None, scene_points: int = 8192,
                    contact_threshold: float = DEFAULT_THRESHOLD_M, mesh_points: int = 1024) -> dict[str, Any]:
    """Convert a ``Clip`` to a dict of fixed-shape tensors.

    ``modalities`` lists the on-request modalities to emit (default: the ones present in
    the clip). ``audio_samples`` fixes the audio length (pad or cut); ``scene_points`` is the
    number of scene points sampled per clip; ``mesh_points`` the surface samples per object.
    """
    a = clip.annotations
    T = a.num_frames
    want = set(modalities) if modalities is not None else {
        k for k, v in (("rgb", clip.frames), ("depth", clip.depth), ("seg", clip.seg), ("audio", clip.audio),
                       ("imu", a.imu), ("gaze", a.gaze_2d), ("scene", clip.scene),
                       ("mesh", clip.meshes)) if v is not None}
    hands = [a.hands[s] for s in HAND_SIDES]
    O = max_objects
    obj_pose = np.zeros((O, T, 4, 4), np.float32)
    obj_box = np.zeros((O, T, 4), np.float32)
    obj_dims = np.zeros((O, 3), np.float32)
    obj_valid = np.zeros((O, T), bool)
    obj_pose_valid = np.zeros((O, T), bool)
    for k, o in enumerate(a.objects[:O]):
        obj_pose[k], obj_box[k], obj_dims[k], obj_valid[k] = o.T_cam_obj, o.bbox_2d, o.dimensions, o.valid
        obj_pose_valid[k] = o.valid & np.isfinite(o.T_cam_obj).all(axis=(1, 2))
    cam = a.T_world_cam if a.T_world_cam is not None else np.full((T, 4, 4), np.nan, np.float32)
    K = clip.K if clip.K is not None else np.full((3, 3), np.nan, np.float32)
    out: dict[str, Any] = {
        "dataset": clip.meta.dataset,
        "sequence_id": clip.meta.sequence_id,
        "frame_indices": torch.from_numpy(a.frame_indices),
        "timestamps": torch.from_numpy(a.timestamps.astype(np.float64)),
        "K": _f32(K),
        "K_valid": torch.tensor(clip.K is not None),
        "hand_valid": torch.from_numpy(np.stack([h.valid for h in hands])),                   # (2, T)
        "hand_joints_2d": _f32(np.stack([h.joints_2d for h in hands])),                        # (2, T, 21, 2)
        "hand_joints_3d": _f32(np.stack([h.joints_3d for h in hands])),                        # (2, T, 21, 3)
        "hand_joints_3d_valid": torch.from_numpy(np.stack([np.isfinite(h.joints_3d).all((1, 2)) for h in hands])),
        "hand_mano": _f32(np.stack([np.concatenate([h.mano_pose, h.mano_betas, h.mano_trans], -1)
                                    for h in hands])),                                         # (2, T, 61)
        "hand_bbox_2d": _f32(np.stack([h.bbox_2d for h in hands])),                            # (2, T, 4)
        "hand_bbox_valid": torch.from_numpy(np.stack([np.isfinite(h.bbox_2d).all(1) for h in hands])),
        "obj_valid": torch.from_numpy(obj_valid),                                              # (O, T)
        "obj_pose_valid": torch.from_numpy(obj_pose_valid),                                    # (O, T)
        "obj_T_cam_obj": _f32(obj_pose),                                                       # (O, T, 4, 4)
        "obj_bbox_2d": _f32(obj_box),                                                          # (O, T, 4)
        "obj_dims": _f32(obj_dims),                                                            # (O, 3)
        "cam_valid": torch.from_numpy(~np.isnan(cam).any(axis=(1, 2))),                        # (T,)
        "T_world_cam": _f32(cam),                                                              # (T, 4, 4)
        "action": a.actions[len(a.actions) // 2].label if a.actions else "",
    }

    def grid_hw() -> tuple[int, int]:
        for x in (clip.frames, clip.depth, clip.seg):
            if x is not None:
                return x.shape[1:3]
        return clip.meta.height, clip.meta.width

    if "rgb" in want and clip.frames is not None:
        out["frames"] = torch.from_numpy(clip.frames).permute(0, 3, 1, 2).float().div_(255)   # (T, 3, H, W)
    if "depth" in want:
        H, W = grid_hw()
        d = clip.depth if clip.depth is not None else np.zeros((T, H, W), np.float32)
        out["depth"] = _f32(d)[:, None]                                                        # (T, 1, H, W) m
        out["depth_valid"] = torch.tensor(clip.depth is not None)
    if "seg" in want:
        H, W = grid_hw()
        s = clip.seg if clip.seg is not None else np.zeros((T, H, W), np.uint8)
        out["seg"] = torch.from_numpy(np.ascontiguousarray(s, np.uint8))                      # (T, H, W)
        out["seg_valid"] = torch.tensor(clip.seg is not None)
    if "audio" in want:
        x = clip.audio if clip.audio is not None else np.zeros((0, 1), np.float32)
        n = audio_samples if audio_samples is not None else len(x)
        buf = np.zeros((n, x.shape[1] if x.size else 1), np.float32)
        buf[:min(n, len(x))] = x[:n]
        out["audio"] = torch.from_numpy(buf.T.copy())                                          # (C, S)
        out["audio_valid"] = torch.tensor(clip.audio is not None)
    if "imu" in want:
        if a.imu is not None:
            v, ok = a.imu.at(a.timestamps)
            # Gravity direction in the camera frame: the mean specific force points up, so
            # down is its negation. Only meaningful when the IMU frame is the camera frame.
            g = -np.nanmean(a.imu.accel, 0) if len(a.imu.t) else np.full(3, np.nan)
            g = g / np.linalg.norm(g) if np.isfinite(g).all() and np.linalg.norm(g) > 0 else np.full(3, np.nan)
            cam_frame = a.imu.frame == "camera"
        else:
            v, ok, g, cam_frame = np.zeros((T, 6), np.float32), np.zeros(T, bool), np.full(3, np.nan), False
        out["imu"] = _f32(v)                                                                   # (T, 6) accel, gyro
        out["imu_valid"] = torch.from_numpy(ok)                                                # (T,)
        out["imu_in_camera_frame"] = torch.tensor(cam_frame)
        out["gravity_cam"] = _f32(g if cam_frame else np.zeros(3))                            # (3,) unit, down
    if "gaze" in want:
        gz = a.gaze_2d if a.gaze_2d is not None else np.full((T, 2), np.nan, np.float32)
        out["gaze_2d"] = _f32(gz)                                                              # (T, 2)
        out["gaze_valid"] = torch.from_numpy(np.isfinite(gz).all(1))                           # (T,)
    if "scene" in want:
        pts = np.zeros((scene_points, 6), np.float32)
        if clip.scene is not None and len(clip.scene.xyz):
            # Deterministic per sequence, so a clip always sees the same subsample.
            rng = np.random.default_rng(zlib.crc32(clip.meta.sequence_id.encode()))
            idx = rng.choice(len(clip.scene.xyz), scene_points, replace=len(clip.scene.xyz) < scene_points)
            pts[:, :3] = clip.scene.xyz[idx]
            pts[:, 3:] = clip.scene.rgb[idx] / 255.0
        out["scene_points"] = torch.from_numpy(pts)                                            # (N, 6) xyz rgb
        out["scene_valid"] = torch.tensor(clip.scene is not None)
    if "mesh" in want:
        pts = np.zeros((O, mesh_points, 6), np.float32)
        ok = np.zeros(O, bool)
        for k, o in enumerate(a.objects[:O]):
            m = (clip.meshes or {}).get(o.object_id)
            if m is None or not len(m.points):
                continue
            idx = np.arange(mesh_points) % len(m.points)  # stored in random order: a prefix is uniform
            pts[k, :, :3], pts[k, :, 3:] = m.points[idx], m.normals[idx]
            ok[k] = True
        out["object_points"] = torch.from_numpy(pts)                                           # (O, P, 6) obj frame
        out["object_points_valid"] = torch.from_numpy(ok)                                      # (O,)
    if "contact" in want:
        c = np.zeros((2, O, T), bool)
        dist = np.full((2, O, T), np.nan, np.float32)
        cc, dd = hand_object_contact(a, contact_threshold, max_objects=O)
        c[:, :cc.shape[1]], dist[:, :dd.shape[1]] = cc, dd
        out["contact"] = torch.from_numpy(c)                                                   # (2, O, T)
        out["contact_dist"] = _f32(dist)                                                       # (2, O, T) m
        out["contact_valid"] = torch.from_numpy(np.isfinite(dist))
    return out


class EgoClipDataset(Dataset):
    def __init__(self, windows: list[ClipWindow], loader: EgoLoader, with_frames: bool | None = None,
                 max_objects: int = 4, audio_samples: int | None = None, scene_points: int = 8192,
                 contact_threshold: float = DEFAULT_THRESHOLD_M, mesh_points: int = 1024):
        self.windows = windows
        self.loader = loader
        if with_frames is not None:  # backward compatibility with the v1 signature
            loader.modalities = loader.modalities | {"rgb"} if with_frames else loader.modalities - {"rgb"}
        self.modalities = set(loader.modalities)
        self.max_objects = max_objects
        self.audio_samples = audio_samples
        self.scene_points = scene_points
        self.contact_threshold = contact_threshold
        self.mesh_points = mesh_points

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        clip = self.loader.load_window(self.windows[i])
        return clip_to_tensors(clip, self.max_objects, self.modalities, self.audio_samples,
                               self.scene_points, self.contact_threshold, self.mesh_points)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "EgoClipDataset":
        """``cfg`` is the ``data`` section of an experiment config (see configs/experiment/).

        ``modalities`` (default ["rgb"]) picks what to load. ``with_frames: false`` is the old
        spelling of leaving "rgb" out. The audio length is fixed to ``clip_len / fps`` seconds
        so mixed-dataset batches collate; that needs ``fps`` to be set.
        """
        mods = list(cfg.get("modalities") or ["rgb"])
        if cfg.get("with_frames") is False:
            mods = [m for m in mods if m != "rgb"]
        mods = sorted(check(mods))
        audio = cfg.get("audio", {}) or {}
        loader = EgoLoader(transform=build_transforms(cfg.get("transforms")),
                           adapter_kwargs=cfg.get("adapter_kwargs", {}), modalities=mods,
                           audio_sr=int(audio.get("sample_rate", 16000)), audio_mono=bool(audio.get("mono", True)))
        windows: list[ClipWindow] = []
        for src in cfg["sources"]:
            idx = loader.index(src["dataset"])
            metas = idx.filter(query=src.get("query"), require=src.get("require", []), **src.get("tags", {}))
            for m in src.get("require_modalities", []):
                metas = [x for x in metas if m in x.modalities]
            if src.get("max_sequences"):
                metas = metas[: int(src["max_sequences"])]
            windows += sample_windows(metas, clip_len=cfg["clip_len"], fps=src.get("fps", cfg.get("fps")),
                                      stride_s=src.get("stride_s"), mode=src.get("sampling", "sliding"),
                                      adapters={src["dataset"]: loader.adapter(src["dataset"])})
        audio_samples = None
        if "audio" in mods and cfg.get("fps"):
            audio_samples = int(round(cfg["clip_len"] / float(cfg["fps"]) * loader.audio_sr))
        return cls(windows, loader, max_objects=cfg.get("max_objects", 4), audio_samples=audio_samples,
                   scene_points=int(cfg.get("scene_points", 8192)),
                   contact_threshold=float(cfg.get("contact_threshold", DEFAULT_THRESHOLD_M)),
                   mesh_points=int(cfg.get("mesh_points", 1024)))
