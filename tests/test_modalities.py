"""Unit tests for the v1.1 modalities. No dataset needed."""

import struct

import numpy as np
import pytest

from egograph.data.contact import box_signed_distance, hand_object_contact
from egograph.data.gpmf import GOPRO_TO_OPENCV, _orin_to_xyz, _walk
from egograph.data.modalities import MODALITIES, check
from egograph.data.schema import (Annotations, Clip, ImuTrack, ObjectTrack, ScenePoints, SequenceMeta)
from egograph.data.transforms import CenterCrop, Resize


def _meta(T=4, H=40, W=60, mods=("rgb", "depth", "seg")):
    return SequenceMeta(dataset="t", sequence_id="s", video_path="v.mp4", fps=10.0, num_frames=T,
                        width=W, height=H, duration_s=T / 10, modalities=list(mods))


def _clip(T=4, H=40, W=60):
    ann = Annotations.empty(np.arange(T), 10.0)
    ann.hands["right"].bbox_2d[:] = [10, 10, 20, 20]
    ann.gaze_2d = np.tile(np.float32([30, 20]), (T, 1))
    ann.imu = ImuTrack(np.linspace(0, 0.3, 61), np.tile(np.float32([0, -9.81, 0]), (61, 1)),
                       np.zeros((61, 3), np.float32), "camera")
    depth = np.full((T, H, W), 0.75, np.float32)
    depth[:, :, :5] = 0
    seg = np.zeros((T, H, W), np.uint8)
    seg[:, 10:20, 10:20] = 2
    rng = np.random.default_rng(0)
    scene = ScenePoints(rng.normal(size=(100, 3)).astype(np.float32), rng.integers(0, 255, (100, 3), dtype=np.uint8),
                        np.zeros(100, np.int32))
    return Clip(_meta(T, H, W), np.zeros((T, H, W, 3), np.uint8), np.float32([[50, 0, 30], [0, 50, 20], [0, 0, 1]]),
                ann, depth=depth, seg=seg, audio=rng.uniform(-1, 1, (1600, 1)).astype(np.float32), audio_sr=4000,
                scene=scene)


def test_registry():
    assert {"rgb", "depth", "seg", "audio", "imu", "gaze", "scene", "contact"} <= set(MODALITIES)
    with pytest.raises(ValueError):
        check(["rgb", "lidar"])


# ---------------------------------------------------------------- GPMF
def _klv(key: bytes, typ: str, size: int, payload: bytes) -> bytes:
    rep = len(payload) // size if size else 0
    head = key + (bytes([0]) if typ == "\0" else typ.encode()) + bytes([size]) + struct.pack(">H", rep)
    return head + payload + b"\0" * ((-len(payload)) % 4)


def test_gpmf_walk_and_orin():
    raw = np.array([[100, 200, 300], [-100, 50, 0]], ">i2")
    strm = (_klv(b"SCAL", "s", 2, struct.pack(">h", 100)) + _klv(b"ORIN", "c", 1, b"YxZ")
            + _klv(b"ACCL", "s", 6, raw.tobytes()))
    devc = _klv(b"DEVC", "\0", 1, _klv(b"STRM", "\0", 1, strm))
    seen = {}
    _walk(devc, lambda k, typ, size, rep, p: seen.__setitem__(k.decode(), (typ, size, rep, p)))
    assert set(seen) == {"SCAL", "ORIN", "ACCL"}
    typ, size, rep, p = seen["ACCL"]
    vals = np.frombuffer(p, ">i2").reshape(rep, 3).astype(np.float32) / 100
    xyz = _orin_to_xyz(vals, "YxZ")  # channel 0 is +Y, channel 1 is -X, channel 2 is +Z
    np.testing.assert_allclose(xyz[0], [-2.0, 1.0, 3.0])
    # GoPro (x right, y up, z back) -> OpenCV (x right, y down, z forward)
    np.testing.assert_allclose(np.float32([0, 9.8, 0]) @ GOPRO_TO_OPENCV, [0, -9.8, 0])


# ---------------------------------------------------------------- IMU / contact
def test_imu_at_and_gap():
    t = np.r_[np.linspace(0, 1, 201), np.linspace(2, 3, 201)]
    imu = ImuTrack(t, np.c_[t, t, t].astype(np.float32), np.zeros((len(t), 3), np.float32))
    v, ok = imu.at(np.array([0.5, 1.5, 2.5, 4.0]))
    assert ok.tolist() == [True, False, True, False]
    np.testing.assert_allclose(v[0, :3], 0.5, atol=1e-5)
    assert np.isnan(v[1]).all()
    w = imu.window(0.2, 0.4)
    assert w.t.min() >= 0.2 and w.t.max() <= 0.4


def test_contact_distance():
    d = box_signed_distance(np.float32([[0, 0, 0], [0.15, 0, 0], [0.1, 0.1, 0]]), np.float32([0.2, 0.2, 0.2]))
    np.testing.assert_allclose(d, [-0.1, 0.05, 0.0], atol=1e-6)
    ann = Annotations.empty(np.arange(2), 10.0)
    h = ann.hands["right"]
    h.valid[:] = True
    h.joints_3d[:] = np.float32([0, 0, 1.0])
    h.joints_3d[1] += 0.5   # second frame: far away
    o = ObjectTrack.empty("o", "mug", 2)
    o.valid[:] = True
    o.T_cam_obj[:] = np.eye(4)
    o.T_cam_obj[:, 2, 3] = 1.05
    o.dimensions = np.float32([0.08, 0.08, 0.08])
    ann.objects.append(o)
    c, dist = hand_object_contact(ann, 0.02)
    assert c.shape == (2, 1, 2) and c[1, 0].tolist() == [True, False]
    assert np.isnan(dist[0]).all()     # left hand missing
    np.testing.assert_allclose(dist[1, 0, 0], 0.01, atol=1e-6)


# ---------------------------------------------------------------- clip I/O, transforms, tensors
def test_clip_roundtrip(tmp_path):
    c = _clip()
    c.save(tmp_path / "c")
    d = Clip.load(tmp_path / "c")
    np.testing.assert_allclose(d.depth, c.depth, atol=1e-3)     # 16-bit mm PNG
    np.testing.assert_array_equal(d.seg, c.seg)
    np.testing.assert_allclose(d.audio, c.audio, atol=1e-6)
    assert d.audio_sr == 4000 and d.scene.xyz.shape == (100, 3)
    np.testing.assert_allclose(d.annotations.gaze_2d, c.annotations.gaze_2d)
    np.testing.assert_allclose(d.annotations.hands["right"].bbox_2d, c.annotations.hands["right"].bbox_2d)
    assert d.annotations.imu.frame == "camera" and len(d.annotations.imu.t) == 61
    assert d.meta.modalities == c.meta.modalities


def test_transforms_keep_modalities_aligned():
    c = Resize(20)(_clip())           # 40x60 -> 20x30
    assert c.depth.shape == (4, 20, 30) and c.seg.shape == (4, 20, 30)
    assert set(np.unique(c.depth)) == {0.0, 0.75}       # nearest: no blended depths
    assert set(np.unique(c.seg)) == {0, 2}
    np.testing.assert_allclose(c.annotations.gaze_2d[0], [(30 + .5) / 2 - .5, (20 + .5) / 2 - .5])
    c = CenterCrop(20, 20)(c)
    assert c.depth.shape == (4, 20, 20) and c.frames.shape == (4, 20, 20, 3)
    np.testing.assert_allclose(c.annotations.hands["right"].bbox_2d[0], np.float32([4.75, 4.75, 9.75, 9.75]) - [5, 0, 5, 0])


def test_tensors_fixed_shapes_and_missing_modalities():
    pytest.importorskip("torch")
    from torch.utils.data import default_collate

    from egograph.data.torch_dataset import clip_to_tensors

    mods = {"rgb", "depth", "seg", "audio", "imu", "gaze", "scene", "contact"}
    full = clip_to_tensors(_clip(), modalities=mods, audio_samples=2000, scene_points=64)
    bare = _clip()
    bare.depth = bare.seg = bare.audio = bare.scene = None
    bare.annotations.imu = bare.annotations.gaze_2d = None
    empty = clip_to_tensors(bare, modalities=mods, audio_samples=2000, scene_points=64)
    assert {k for k in full} == {k for k in empty}
    b = default_collate([full, empty])
    assert b["depth"].shape == (2, 4, 1, 40, 60) and b["depth_valid"].tolist() == [True, False]
    assert b["audio"].shape == (2, 1, 2000) and b["audio_valid"].tolist() == [True, False]
    assert b["scene_points"].shape == (2, 64, 6)
    assert b["imu_valid"][0].all() and not b["imu_valid"][1].any()
    np.testing.assert_allclose(b["gravity_cam"][0].numpy(), [0, 1, 0], atol=1e-6)  # down = +y in OpenCV
    assert b["gaze_valid"][0].all() and not b["gaze_valid"][1].any()
    assert b["contact"].shape == (2, 2, 4, 4)
