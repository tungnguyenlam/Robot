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


# ---------------------------------------------------------------- v1.1 modalities on real data
@require_index("hoi4d")
def test_hoi4d_depth_seg_scene():
    L = EgoLoader(modalities=["depth", "seg", "scene"])
    idx = L.index("hoi4d")
    m = next(x for x in idx.metas if {"depth", "hand_pose"} <= set(x.modalities))
    c = L.load_clip("hoi4d", m.sequence_id, [150])
    assert c.depth.shape == (1, m.height, m.width) and c.seg.shape == (1, m.height, m.width)
    h = c.annotations.hands["right"] if c.annotations.hands["right"].valid[0] else c.annotations.hands["left"]
    assert h.valid[0]
    uv = np.round(h.joints_2d[0]).astype(int)
    uv[:, 0] = uv[:, 0].clip(0, m.width - 1)
    uv[:, 1] = uv[:, 1].clip(0, m.height - 1)
    z = c.depth[0][uv[:, 1], uv[:, 0]]
    # Depth at the 2D keypoints matches the 3D joint depth (pixel-aligned, metres).
    assert abs(np.median(z[z > 0]) - np.median(h.joints_3d[0][:, 2])) < 0.08
    assert (c.seg[0][uv[:, 1], uv[:, 0]] > 0).mean() > 0.5   # keypoints fall on a mask

    # The scene cloud lives in the T_world_cam frame: projected into a view, it agrees with depth.
    # About 1 in 5 HOI4D sequences has a scene/pose registration error of 10-60 cm in the source
    # data (docs/MODALITIES.md), so require most of a few sequences, not every one, to agree.
    def gap(m):
        c = L.load_clip("hoi4d", m.sequence_id, [150], with_frames=False)
        Tcw = np.linalg.inv(c.annotations.T_world_cam[0])
        p = c.scene.xyz @ Tcw[:3, :3].T + Tcw[:3, 3]
        p = p[p[:, 2] > 0.1]
        u = (m.K[0, 0] * p[:, 0] / p[:, 2] + m.K[0, 2]).astype(int)
        v = (m.K[1, 1] * p[:, 1] / p[:, 2] + m.K[1, 2]).astype(int)
        ok = (u >= 0) & (u < m.width) & (v >= 0) & (v < m.height)
        D = c.depth[0][v[ok], u[ok]]
        good = D > 0
        return np.median(np.abs(D[good] - p[ok][good, 2]))

    seqs = [x for x in idx.metas if {"depth", "scene"} <= set(x.modalities)][::400][:6]
    gaps = np.array([gap(x) for x in seqs])
    assert (gaps < 0.05).sum() >= 4, gaps


@require_index("epic_kitchens")
def test_epic_audio_imu():
    L = EgoLoader(modalities=["audio", "imu"])
    idx = L.index("epic_kitchens")
    m = next(x for x in idx.metas if "imu" in x.modalities)
    c = L.load_clip("epic_kitchens", m.sequence_id, L.frames_at(m, 30.0, 8, 30))
    ts = c.annotations.timestamps
    span = ts[-1] - ts[0] + np.median(np.diff(ts))        # first frame to one period after the last
    assert c.audio.shape == (round(span * 16000), 1) and c.audio_sr == 16000
    imu = c.annotations.imu
    assert imu.frame == "camera" and len(imu.t) > 20
    v, ok = imu.at(c.annotations.timestamps)
    assert ok.all() and 7 < np.linalg.norm(v[:, :3].mean(0)) < 12   # gravity dominates
    assert v[:, 1].mean() < 0   # specific force points up = -y in the OpenCV frame


@require_index("ego4d")
def test_ego4d_fho_boxes():
    from egograph.paths import cache_root
    import json

    L = EgoLoader(modalities=["rgb"])
    idx = L.index("ego4d")
    m = next(x for x in idx.metas if "hand_box" in x.modalities)
    acts = json.loads((cache_root("ego4d") / "ann" / f"{m.sequence_id}.json").read_text())["actions"]
    b = next(b for a in acts for b in a.get("boxes", []) if b["type"].endswith("_hand"))
    c = L.load_clip("ego4d", m.sequence_id, [b["frame"] - 1, b["frame"], b["frame"] + 1], with_frames=False)
    h = c.annotations.hands[b["type"].split("_")[0]]
    np.testing.assert_allclose(h.bbox_2d[1], b["xyxy"], atol=1e-3)
    assert np.isnan(h.bbox_2d[0]).all() and np.isnan(h.bbox_2d[2]).all()   # sparse: key frames only
    x0, y0, x1, y1 = b["xyxy"]
    assert 0 <= x0 < x1 <= m.width + 2 and 0 <= y0 < y1 <= m.height + 2
