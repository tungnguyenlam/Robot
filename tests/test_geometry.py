import numpy as np

from egograph.data.geometry import box_corners, project, scale_intrinsics, transform_points
from egograph.data.schema import Annotations, Clip, SequenceMeta
from egograph.data.transforms import CenterCrop, Compose, Resize


def test_transforms_keep_projection_consistent():
    """After resizing and cropping, projecting 3D joints with the new K must still land on the
    transformed 2D joints."""
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], np.float32)
    meta = SequenceMeta("toy", "s", "v", 30, 10, 1920, 1080, 1, intrinsics=K.tolist())
    ann = Annotations.empty(np.arange(2), 30)
    j3d = np.random.default_rng(0).uniform([-0.2, -0.1, 0.4], [0.2, 0.1, 0.8], (2, 21, 3)).astype(np.float32)
    h = ann.hands["left"]
    h.joints_3d[:] = j3d
    h.joints_2d[:] = project(j3d, K)
    h.valid[:] = True
    clip = Clip(meta, np.zeros((2, 1080, 1920, 3), np.uint8), K, ann)
    clip = Compose([Resize(256), CenterCrop(256, 256)])(clip)
    assert clip.frames.shape == (2, 256, 256, 3)
    np.testing.assert_allclose(project(h.joints_3d, clip.K), clip.annotations.hands["left"].joints_2d, atol=0.05)


def test_scale_intrinsics_pixel_centres():
    K = np.array([[100, 0, 49.5], [0, 100, 49.5], [0, 0, 1]], np.float32)  # centre of a 100 px image
    K2 = scale_intrinsics(K, 0.5, 0.5)
    assert np.isclose(K2[0, 2], 24.5)  # centre of a 50 px image


def test_box_corners_transform():
    c = box_corners(np.array([2, 4, 6]))
    assert c.shape == (8, 3) and np.allclose(c.max(0), [1, 2, 3])
    T = np.eye(4)
    T[:3, 3] = [0, 0, 5]
    assert np.allclose(transform_points(T, c).mean(0), [0, 0, 5])
