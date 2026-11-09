"""MANO forward kinematics producing joints in the standard 21-joint OpenPose order.

This wraps ``smplx.MANO`` and is equivalent to manopth with ``use_pca=False``,
``flat_hand_mean=True`` (the egoformat convention), and output in meters. Fingertips come from the usual MANO
vertex ids.
"""

from __future__ import annotations

import numpy as np

from egograph.paths import mano_dir

# smplx MANO gives 16 joints (wrist, then index, middle, pinky, ring, thumb with 3 each).
# Appending the fingertip vertices (thumb, index, middle, ring, pinky) gives 21 joints.
# REORDER maps that layout to OpenPose order, using the same table as manopth.
TIP_VERTS = (745, 317, 444, 556, 673)
REORDER = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)


class ManoModel:
    def __init__(self, model_dir: str | None = None, device: str = "cpu", flat_hand_mean: bool = True,
                 fix_left_shapedirs: bool = False):
        import smplx
        import torch

        self.torch = torch
        self.device = device
        root = model_dir or str(mano_dir())
        self.layers = {
            side: smplx.MANO(root, is_rhand=(side == "right"), use_pca=False,
                             flat_hand_mean=flat_hand_mean).to(device).eval()
            for side in ("left", "right")
        }
        # The released MANO_LEFT.pkl has the x component of its shapedirs flipped. smplx users
        # often "fix" this, but HOI4D (like manopth-based datasets) was fitted with the
        # original model. With the fix, HOI4D left hands reproject 29 px off; without it, 0.85 px.
        # So the egoformat convention is no fix, and fix_left_shapedirs is opt-in.
        l, r = self.layers["left"], self.layers["right"]
        if fix_left_shapedirs and torch.sum(torch.abs(l.shapedirs[:, 0, :] - r.shapedirs[:, 0, :])) < 1:
            l.shapedirs[:, 0, :] *= -1

    def forward(self, side: str, pose: np.ndarray, betas: np.ndarray, trans: np.ndarray):
        """pose (N, 48), betas (N, 10), trans (N, 3) -> (joints (N, 21, 3), vertices (N, 778, 3))."""
        torch = self.torch
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)  # noqa: E731
        with torch.no_grad():
            out = self.layers[side](global_orient=t(pose)[:, :3], hand_pose=t(pose)[:, 3:],
                                    betas=t(betas), transl=t(trans))
            verts = out.vertices
            joints = torch.cat([out.joints[:, :16], verts[:, list(TIP_VERTS)]], dim=1)[:, list(REORDER)]
        return joints.cpu().numpy(), verts.cpu().numpy()

    def joints(self, side: str, pose, betas, trans) -> np.ndarray:
        return self.forward(side, pose, betas, trans)[0]

    def faces(self, side: str) -> np.ndarray:
        return np.asarray(self.layers[side].faces, dtype=np.int64)
