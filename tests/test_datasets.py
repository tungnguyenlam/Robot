"""Data-dependent checks. They run on amdserver and skip elsewhere."""

import numpy as np
import pytest

from conftest import require_index
from egograph.data import EgoLoader
from egograph.data.geometry import box_corners, project, transform_points
from egograph.data.loader import sample_windows


@pytest.fixture(scope="module")
def loader():
    return EgoLoader()


@require_index("hoi4d")
def test_hoi4d_conventions(loader):
    """Projected 3D object boxes must match the annotated 2D boxes, and projected MANO joints
    must match the annotated keypoints. This pins the Euler-order and flat_hand_mean choices."""
    idx = loader.index("hoi4d")
    rng = np.random.default_rng(0)
    metas = idx.filter(require=["has_hand_pose"])
    ious, errs = [], {"left": [], "right": []}
    for m in rng.choice(metas, 12, replace=False):
        ann = loader.adapter("hoi4d").load_annotations(m, [0, 100, 200])
        for o in ann.objects:
            for t in np.flatnonzero(o.valid & np.isfinite(o.bbox_2d).all(1)):
                uv = project(transform_points(o.T_cam_obj[t], box_corners(o.dimensions)), m.K)
                a, b = [*uv.min(0), *uv.max(0)], o.bbox_2d[t]
                inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                ious.append(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter))
        for s, h in ann.hands.items():
            if h.valid.any():
                errs[s].append(np.linalg.norm(project(h.joints_3d[h.valid], m.K) - h.joints_2d[h.valid], axis=-1).mean())
    assert np.mean(ious) > 0.85, np.mean(ious)
    for s, e in errs.items():
        if e:
            assert np.median(e) < 3.0, (s, np.median(e))


@require_index("hoi4d")
def test_hoi4d_clip(loader):
    m = loader.index("hoi4d").filter(require=["has_hand_pose"])[0]
    clip = loader.load_clip("hoi4d", m.sequence_id, loader.frames_at(m, 1.0, 8, 15))
    assert clip.frames.shape == (8, m.height, m.width, 3)
    a = clip.annotations
    assert a.T_world_cam is not None and np.isfinite(a.T_world_cam).all()
    assert any(h.valid.all() for h in a.hands.values())
    assert a.objects and a.objects[0].valid.all()


@require_index("epic_kitchens")
def test_epic_actions_and_windows(loader):
    m = loader.index("epic_kitchens")["P01_01"]
    acts = loader.adapter("epic_kitchens").actions(m)
    assert len(acts) > 50 and acts[0].label == "open door"
    ws = sample_windows([m], 16, fps=30, mode="actions", adapters={"epic_kitchens": loader.adapter("epic_kitchens")})
    assert len(ws) == len(acts)
    clip = loader.load_window(ws[10])
    assert clip.frames.shape[0] == 16 and clip.K is None
    assert any(a.label == ws[10].label for a in clip.annotations.actions)


@require_index("hoi4d")
def test_torch_dataset_batches(loader):
    torch = pytest.importorskip("torch")
    from egograph.config import load_config
    from egograph.data.torch_dataset import EgoClipDataset
    from egograph.paths import REPO_ROOT

    cfg = load_config(REPO_ROOT / "configs" / "experiment" / "example_dataloader.yaml",
                      ["data.sources.0.max_sequences=2", "data.sources.1.max_sequences=1"])
    ds = EgoClipDataset.from_config(cfg["data"])
    assert len(ds) > 0
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    b = next(iter(dl))
    T, S = cfg["data"]["clip_len"], cfg["data"]["transforms"]["resize"]
    assert b["frames"].shape == (2, T, 3, S, S)
    assert b["hand_joints_2d"].shape == (2, 2, T, 21, 2)
