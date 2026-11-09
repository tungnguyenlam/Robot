import numpy as np

from egograph.data.schema import ActionSegment, Annotations, Clip, ObjectTrack, SequenceMeta


def _clip(T=5):
    meta = SequenceMeta("toy", "seq-1", "v.mp4", 15.0, 100, 64, 48, 6.6, intrinsics=np.eye(3).tolist())
    ann = Annotations.empty(np.arange(T) * 2, meta.fps)
    ann.hands["right"].valid[:] = True
    ann.hands["right"].joints_2d[:] = np.random.rand(T, 21, 2)
    o = ObjectTrack.empty("1", "mug", T)
    o.valid[1:] = True
    o.T_cam_obj[1:] = np.eye(4)
    o.dimensions = np.array([0.1, 0.1, 0.2], np.float32)
    ann.objects = [o]
    ann.T_world_cam = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    ann.actions = [ActionSegment(0.0, 0.5, "grasp", verb="grasp")]
    frames = np.random.randint(0, 255, (T, 48, 64, 3), np.uint8)
    return Clip(meta, frames, meta.K, ann)


def test_roundtrip(tmp_path):
    c = _clip()
    c.save(tmp_path / "c", frame_format="png")
    d = Clip.load(tmp_path / "c")
    assert np.array_equal(d.frames, c.frames)
    a, b = c.annotations, d.annotations
    assert np.array_equal(a.frame_indices, b.frame_indices)
    np.testing.assert_allclose(b.hands["right"].joints_2d, a.hands["right"].joints_2d)
    assert np.isnan(b.hands["left"].joints_3d).all() and not b.hands["left"].valid.any()
    assert b.objects[0].label == "mug" and b.objects[0].valid.tolist() == [False, True, True, True, True]
    assert b.actions[0].label == "grasp"
    assert d.meta == c.meta
