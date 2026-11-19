"""Object meshes: OBJ reading, simplification for drawing, surface sampling, and a disk cache.

The HOI4D CAD models are 50k-100k triangles each, too heavy to parse per clip or to send to the
browser. ``load_mesh`` parses a model once and caches an ``ObjectMesh`` as .npz under
``$EGO_INDEX_ROOT/cache/<dataset>/meshes/``: a simplified mesh (vertex clustering, about
``DRAW_FACES`` triangles) for drawing, and ``SURFACE_POINTS`` area-weighted surface samples with
normals from the full-resolution mesh, for training (penetration and contact terms).
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import numpy as np

from robot.data.schema import ObjectMesh
from robot.paths import cache_root

DRAW_FACES = 3000
SURFACE_POINTS = 8192
_CACHE_VERSION = 1


def read_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(V,3) float32 vertices and (F,3) int32 triangles of a Wavefront OBJ. Polygons are
    fan-triangulated; texture and normal indices are ignored."""
    vs: list[list[str]] = []
    fs: list[list[int]] = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vs.append(line.split()[1:4])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    fs.append([idx[0], idx[k], idx[k + 1]])
    v = np.asarray(vs, np.float32).reshape(-1, 3)
    f = np.asarray(fs, np.int64).reshape(-1, 3)
    f = np.where(f < 0, f + len(v), f - 1)  # OBJ is 1-based; negative indices count from the end
    return v, f.astype(np.int32)


def face_areas_normals(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    a = np.linalg.norm(n, axis=1)
    return 0.5 * a, n / np.maximum(a, 1e-12)[:, None]


def sample_surface(v: np.ndarray, f: np.ndarray, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """n area-weighted points on the surface and their face normals."""
    area, normal = face_areas_normals(v, f)
    rng = np.random.default_rng(seed)
    face = rng.choice(len(f), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w = np.stack([1 - s, s * (1 - r2), s * r2], 1).astype(np.float32)
    p = (v[f[face]] * w[:, :, None]).sum(1)
    return p.astype(np.float32), normal[face].astype(np.float32)


def simplify(v: np.ndarray, f: np.ndarray, target_faces: int = DRAW_FACES) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-clustering simplification: snap vertices to a grid, merge each cell into its mean
    and drop collapsed or duplicate triangles. The grid is refined until the face count is
    at most ``target_faces``. Good enough for an overlay; not watertight-preserving."""
    if len(f) <= target_faces:
        return v, f
    lo, ext = v.min(0), np.ptp(v, 0).max()
    best = (v, f)
    for cells in (128, 96, 64, 48, 32, 24, 16, 12, 8):
        key = np.floor((v - lo) / (ext / cells + 1e-12)).astype(np.int64)
        _, cid, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        cid = cid.reshape(-1)
        nv = np.zeros((len(counts), 3), np.float64)
        np.add.at(nv, cid, v)
        nv = (nv / counts[:, None]).astype(np.float32)
        nf = cid[f]
        nf = nf[(nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])]
        _, first = np.unique(np.sort(nf, 1), axis=0, return_index=True)
        nf = nf[np.sort(first)]  # drop duplicates, keep winding and order
        best = (nv, nf.astype(np.int32))
        if len(nf) <= target_faces:
            break
    return best


def build_mesh(path: str | Path, source: str = "", draw_faces: int = DRAW_FACES,
               surface_points: int = SURFACE_POINTS) -> ObjectMesh:
    v, f = read_obj(path)
    pts, nrm = sample_surface(v, f, surface_points)
    sv, sf = simplify(v, f, draw_faces)
    return ObjectMesh(sv, sf, pts, nrm, source)


def load_mesh(path: str | Path, dataset: str, source: str = "") -> ObjectMesh:
    """Cached ``build_mesh``. The cache key covers the path, size and mtime of the source."""
    path = Path(path)
    st = path.stat()
    key = hashlib.sha1(f"{_CACHE_VERSION}|{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    cache = cache_root(dataset) / "meshes" / f"{path.stem}-{key}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return ObjectMesh(z["vertices"], z["faces"], z["points"], z["normals"], str(z["source"]))
    m = build_mesh(path, source)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".{os.getpid()}-{threading.get_ident()}.tmp.npz")  # unique per writer
    np.savez_compressed(tmp, vertices=m.vertices, faces=m.faces, points=m.points, normals=m.normals,
                        source=np.array(source))
    tmp.replace(cache)
    return m

