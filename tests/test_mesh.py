import numpy as np
import pytest

from conftest import require_index
from robot.data.mesh import load_mesh, read_obj, sample_surface, simplify


def _cube_obj(path, s=0.1):
    v = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)]) / 2
    quads = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]
    path.write_text("".join(f"v {a} {b} {c}\n" for a, b, c in v)
                    + "".join(f"f {a}/1 {b}/1 {c}/1 {d}/1\n" for a, b, c, d in quads))


def test_obj_sampling_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("EGO_CACHE_ROOT", str(tmp_path / "idx"))
    p = tmp_path / "cube.obj"
    _cube_obj(p)
    v, f = read_obj(p)
    assert v.shape == (8, 3) and f.shape == (12, 3) and f.min() == 0 and f.max() == 7  # quads fan-split
    pts, nrm = sample_surface(v, f, 2000)
    assert np.allclose(np.abs(pts).max(1), 0.05, atol=1e-6)            # every sample on a face
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1, atol=1e-5)
    assert simplify(v, f, 100)[1].shape == f.shape                      # already small: unchanged
    m1 = load_mesh(p, "test", "cube.obj")
    m2 = load_mesh(p, "test", "cube.obj")                               # from the cache
    assert np.array_equal(m1.points, m2.points) and m2.source == "cube.obj"
    assert any((tmp_path / "idx").rglob("cube-*.npz"))


@require_index("hoi4d")
def test_hoi4d_meshes():
    from robot.data import EgoLoader

    L = EgoLoader(modalities=["mesh"], adapter_kwargs={"hoi4d": {"mano": False}})
    idx = L.index("hoi4d")
    for rigid in (True, False):
        m = next(x for x in idx.metas if "mesh" in x.modalities and x.tags["rigid"] == rigid
                 and x.tags["category_name"] in ("Mug", "Laptop"))
        c = L.load_clip("hoi4d", m.sequence_id, [0, 10], with_frames=False)
        assert c.meshes and set(c.meshes) <= {o.object_id for o in c.annotations.objects}
        for o in c.annotations.objects:
            mesh = c.meshes[o.object_id]
            # Each mesh is in its object (part) frame: centred, and its extents match the box dims.
            ext = np.ptp(mesh.vertices, 0)
            assert np.abs(ext - o.dimensions).max() < 0.05, (o.label, ext, o.dimensions)
            assert np.abs(mesh.vertices.min(0) + mesh.vertices.max(0)).max() < 0.02
        if not rigid:   # the laptop's two parts get different meshes
            assert len({o.mesh_path for o in c.annotations.objects}) == len(c.annotations.objects) == 2


@require_index("hoi4d")
def test_mesh_tensors_and_clip_roundtrip(tmp_path):
    pytest.importorskip("torch")
    from robot.data import EgoLoader
    from robot.data.schema import Clip
    from robot.data.torch_dataset import clip_to_tensors

    L = EgoLoader(modalities=["mesh"], adapter_kwargs={"hoi4d": {"mano": False}})
    m = next(x for x in L.index("hoi4d").metas if "mesh" in x.modalities and not x.tags["rigid"])
    c = L.load_clip("hoi4d", m.sequence_id, [0, 1], with_frames=False)
    out = clip_to_tensors(c, max_objects=4, modalities={"mesh"}, mesh_points=256)
    n = len(c.annotations.objects)
    assert out["object_points"].shape == (4, 256, 6)
    assert out["object_points_valid"].tolist() == [True] * n + [False] * (4 - n)
    c.save(tmp_path / "clip")
    back = Clip.load(tmp_path / "clip")
    assert set(back.meshes) == set(c.meshes)
    k = next(iter(c.meshes))
    assert np.array_equal(back.meshes[k].faces, c.meshes[k].faces) and back.meshes[k].source == c.meshes[k].source
