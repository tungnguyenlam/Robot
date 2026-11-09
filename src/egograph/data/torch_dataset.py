"""PyTorch ``Dataset`` over standard clips. It needs the ``torch`` extra.

    from egograph.data.torch_dataset import EgoClipDataset
    ds = EgoClipDataset.from_config(load_config("configs/experiment/example_dataloader.yaml")["data"])
    batch = next(iter(DataLoader(ds, batch_size=4, num_workers=4)))
    batch["frames"].shape  # (B, T, 3, H, W)

Every sample has the same keys and shapes, so the default collate works. Objects are
padded to ``max_objects``, and missing values become 0 with ``*_valid`` set to False.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from egograph.data.loader import ClipWindow, EgoLoader, sample_windows
from egograph.data.schema import HAND_SIDES, Clip
from egograph.data.transforms import build_transforms


def clip_to_tensors(clip: Clip, max_objects: int = 4) -> dict[str, Any]:
    a = clip.annotations
    T = a.num_frames
    f32 = lambda x: torch.from_numpy(np.nan_to_num(np.asarray(x, np.float32)))  # noqa: E731
    hands = [a.hands[s] for s in HAND_SIDES]
    obj_pose = np.zeros((max_objects, T, 4, 4), np.float32)
    obj_box = np.zeros((max_objects, T, 4), np.float32)
    obj_dims = np.zeros((max_objects, 3), np.float32)
    obj_valid = np.zeros((max_objects, T), bool)
    for k, o in enumerate(a.objects[:max_objects]):
        obj_pose[k], obj_box[k], obj_dims[k], obj_valid[k] = o.T_cam_obj, o.bbox_2d, o.dimensions, o.valid
    cam = a.T_world_cam if a.T_world_cam is not None else np.full((T, 4, 4), np.nan, np.float32)
    K = clip.K if clip.K is not None else np.full((3, 3), np.nan, np.float32)
    out: dict[str, Any] = {
        "dataset": clip.meta.dataset,
        "sequence_id": clip.meta.sequence_id,
        "frame_indices": torch.from_numpy(a.frame_indices),
        "timestamps": torch.from_numpy(a.timestamps.astype(np.float64)),
        "K": f32(K),
        "K_valid": torch.tensor(clip.K is not None),
        "hand_valid": torch.from_numpy(np.stack([h.valid for h in hands])),          # (2, T)
        "hand_joints_2d": f32(np.stack([h.joints_2d for h in hands])),                # (2, T, 21, 2)
        "hand_joints_3d": f32(np.stack([h.joints_3d for h in hands])),                # (2, T, 21, 3)
        "hand_mano": f32(np.stack([np.concatenate([h.mano_pose, h.mano_betas, h.mano_trans], -1)
                                   for h in hands])),                                 # (2, T, 61)
        "obj_valid": torch.from_numpy(obj_valid),                                     # (O, T)
        "obj_T_cam_obj": f32(obj_pose),                                               # (O, T, 4, 4)
        "obj_bbox_2d": f32(obj_box),                                                  # (O, T, 4)
        "obj_dims": f32(obj_dims),                                                    # (O, 3)
        "cam_valid": torch.from_numpy(~np.isnan(cam).any(axis=(1, 2))),               # (T,)
        "T_world_cam": f32(cam),                                                      # (T, 4, 4)
        "action": a.actions[len(a.actions) // 2].label if a.actions else "",
    }
    if clip.frames is not None:
        out["frames"] = torch.from_numpy(clip.frames).permute(0, 3, 1, 2).float().div_(255)  # (T, 3, H, W)
    return out


class EgoClipDataset(Dataset):
    def __init__(self, windows: list[ClipWindow], loader: EgoLoader, with_frames: bool = True,
                 max_objects: int = 4):
        self.windows = windows
        self.loader = loader
        self.with_frames = with_frames
        self.max_objects = max_objects

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return clip_to_tensors(self.loader.load_window(self.windows[i], self.with_frames), self.max_objects)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "EgoClipDataset":
        """``cfg`` is the ``data`` section of an experiment config (see configs/experiment/)."""
        loader = EgoLoader(transform=build_transforms(cfg.get("transforms")),
                           adapter_kwargs=cfg.get("adapter_kwargs", {}))
        windows: list[ClipWindow] = []
        for src in cfg["sources"]:
            idx = loader.index(src["dataset"])
            metas = idx.filter(query=src.get("query"), require=src.get("require", []), **src.get("tags", {}))
            if src.get("max_sequences"):
                metas = metas[: int(src["max_sequences"])]
            windows += sample_windows(metas, clip_len=cfg["clip_len"], fps=src.get("fps", cfg.get("fps")),
                                      stride_s=src.get("stride_s"), mode=src.get("sampling", "sliding"),
                                      adapters={src["dataset"]: loader.adapter(src["dataset"])})
        return cls(windows, loader, with_frames=cfg.get("with_frames", True),
                   max_objects=cfg.get("max_objects", 4))
