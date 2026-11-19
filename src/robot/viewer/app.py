"""Dataset viewer: a FastAPI backend over the same adapters and loader used for training.

Every overlay in the browser (hand keypoints, projected MANO joints, 3D object boxes,
camera trajectory, object meshes) is computed by ``EgoLoader`` and ``robot.data.geometry``. Looking at
the viewer therefore also checks the data pipeline.

Run on amdserver, where the data lives:   robot viewer --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import io
import math
import mimetypes
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from robot.data.adapters import ADAPTERS
from robot.data.contact import DEFAULT_THRESHOLD_M, hand_object_distance
from robot.data.geometry import BOX_EDGES, box_corners, project, transform_points
from robot.data.index import SequenceIndex, index_path
from robot.data.loader import EgoLoader, frame_indices_at
from robot.data.schema import (CONTACT_NA, CONTACT_NONE, CONTACT_UNSEGMENTED, CONTACT_UNSURE, HAND_PARENTS,
                                  HAND_SIDES)
from robot.data.video import VideoReader

STATIC = Path(__file__).parent / "static"
FACET_TAGS = ("participant", "split", "category_name", "human", "task", "scenario", "device")
MAX_ANN_FRAMES = 900
MANO_VERTS = 778


def _clean(x: Any) -> Any:
    """Make numpy objects JSON-safe. NaN and inf become null, and floats are rounded."""
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(x) else round(float(x), 3)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


class State:
    def __init__(self) -> None:
        self.loader = EgoLoader()
        self._lock = threading.Lock()
        self._mtimes: dict[str, float] = {}

    def index(self, dataset: str) -> SequenceIndex | None:
        """Return the index, reloading it automatically after ``robot index`` rewrites it."""
        p = index_path(dataset)
        if not p.exists():
            return None
        with self._lock:
            m = p.stat().st_mtime
            if self._mtimes.get(dataset) != m:
                self.loader._indexes[dataset] = SequenceIndex.load(dataset)
                self._mtimes[dataset] = m
            return self.loader._indexes[dataset]

    def meta(self, dataset: str, sid: str):
        idx = self.index(dataset)
        if idx is None or sid not in idx:
            raise HTTPException(404, f"unknown sequence {dataset}/{sid}")
        return idx[sid]


def create_app() -> FastAPI:
    app = FastAPI(title="robot viewer")
    app.add_middleware(GZipMiddleware, minimum_size=4096)  # chunks with VISOR polygons are large
    st = State()

    # ------------------------------------------------------------------ datasets
    @app.get("/api/datasets")
    def datasets() -> Any:
        out = []
        for name, cls in ADAPTERS.items():
            ad = st.loader.adapter(name)
            idx = st.index(name)
            info = {"name": name, "display_name": cls.display_name, "description": cls.description,
                    "status": ad.status(), "indexed": idx is not None}
            if idx is not None:
                ms = idx.metas
                info.update(
                    sequences=len(ms),
                    hours=sum(m.duration_s for m in ms) / 3600,
                    frames=sum(m.num_frames for m in ms),
                    with_hand_pose=sum(m.has_hand_pose for m in ms),
                    with_object_pose=sum(m.has_object_pose for m in ms),
                    with_camera_pose=sum(m.has_camera_pose for m in ms),
                    with_actions=sum(m.has_actions for m in ms),
                    modalities=Counter(x for m in ms for x in m.modalities).most_common(),
                    facets={t: Counter(str(m.tags[t]) for m in ms if t in m.tags).most_common(60)
                            for t in FACET_TAGS if any(t in m.tags for m in ms[:50])},
                )
            out.append(info)
        return _clean(out)

    @app.get("/api/sequences")
    def sequences(dataset: str, q: str | None = None, require: list[str] = Query(default=[]),
                  tag: list[str] = Query(default=[]), modality: list[str] = Query(default=[]),
                  offset: int = 0, limit: int = 100) -> Any:
        idx = st.index(dataset)
        if idx is None:
            return {"total": 0, "items": []}
        tags = dict(t.split(":", 1) for t in tag if ":" in t)
        ms = [m for m in idx.filter(query=q, require=require)
              if all(str(m.tags.get(k)) == v for k, v in tags.items())
              and all(x in m.modalities for x in modality)]
        return _clean({"total": len(ms), "items": [m.to_dict() for m in ms[offset:offset + limit]]})

    @app.get("/api/sequence/{dataset}/{sid}")
    def sequence(dataset: str, sid: str) -> Any:
        meta = st.meta(dataset, sid)
        ad = st.loader.adapter(dataset)
        acts = ad.actions(meta)
        traj = None
        if meta.has_camera_pose and hasattr(ad, "camera_poses"):
            poses = ad.camera_poses(meta)
            traj = [[int(k), *v[:3, 3].tolist()] for k, v in sorted(poses.items())]
        return _clean({"meta": meta.to_dict(), "actions": [a.__dict__ for a in acts],
                       "camera_trajectory": traj, "video_url": f"/video/{dataset}/{sid}",
                       "hand_parents": HAND_PARENTS, "box_edges": BOX_EDGES,
                       "polygons": _polygon_summary(dataset, sid)})

    # ------------------------------------------------------------------ polygon masks (VISOR)
    def _polygon_masks(dataset: str, sid: str):
        meta = st.meta(dataset, sid)
        return st.loader.adapter(dataset).load_polygon_masks(meta) if "seg_sparse" in meta.modalities else None

    def _polygon_summary(dataset: str, sid: str) -> dict | None:
        """Entity names, manual frames, and interpolated runs [first, last] for the timeline."""
        pm = _polygon_masks(dataset, sid)
        if pm is None:
            return None
        interp = pm.frames[~pm.manual]
        runs = []
        if len(interp):
            cut = np.flatnonzero(np.diff(interp) > 1)
            runs = np.stack([interp[np.r_[0, cut + 1]], interp[np.r_[cut, len(interp) - 1]]], 1)
        return {"names": pm.names, "manual": pm.frames[pm.manual], "interpolated": runs,
                "contact_codes": {"na": CONTACT_NA, "none": CONTACT_NONE,
                                  "unsegmented": CONTACT_UNSEGMENTED, "unsure": CONTACT_UNSURE}}

    def _polygon_chunk(dataset: str, sid: str, start: int, end: int) -> list | None:
        """Masks of the frames in [start, end): [[i - start, manual, [[entity, contact,
        [x0, y0, x1, y1, ...], ...polygons], ...masks]], ...] in native pixels."""
        pm = _polygon_masks(dataset, sid)
        if pm is None:
            return None
        return [[f["frame"] - start, f["manual"],
                 [[m["entity"], m["contact"], *(p.reshape(-1).tolist() for p in m["polygons"])]
                  for m in f["masks"]]]
                for f in pm.window(start, end)]

    # ------------------------------------------------------------------ annotations
    @lru_cache(maxsize=32)
    def _raw_annotations(dataset: str, sid: str, start: int, end: int):
        """The adapter's ``Annotations`` for one chunk, shared by the JSON and hand-mesh endpoints."""
        return st.loader.adapter(dataset).load_annotations(st.meta(dataset, sid), np.arange(start, end))

    @lru_cache(maxsize=64)
    def _annotations(dataset: str, sid: str, start: int, end: int) -> dict:
        meta = st.meta(dataset, sid)
        fi = np.arange(start, end)
        ann = _raw_annotations(dataset, sid, start, end)
        K = meta.K
        frames: dict[str, Any] = {}
        hands = {}
        for side in HAND_SIDES:
            h = ann.hands[side]
            has_box = np.isfinite(h.bbox_2d).all(1)
            if not h.valid.any() and not has_box.any():
                continue
            mano2d = project(h.joints_3d, K) if K is not None and h.valid.any() else None
            hands[side] = {"valid": h.valid, "joints_2d": h.joints_2d,
                           "mano_2d": mano2d if mano2d is not None and np.isfinite(mano2d).any() else None,
                           "bbox_2d": h.bbox_2d if has_box.any() else None}
        objects = []
        for o in ann.objects:
            corners = np.full((len(fi), 8, 2), np.nan, np.float32)
            axes = np.full((len(fi), 4, 2), np.nan, np.float32)
            if K is not None:
                c = box_corners(o.dimensions)
                L = float(np.nanmax(o.dimensions)) * 0.6 if np.isfinite(o.dimensions).any() else 0.1
                ax = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]], np.float32)
                for t in np.flatnonzero(o.valid):
                    corners[t] = project(transform_points(o.T_cam_obj[t], c), K)
                    axes[t] = project(transform_points(o.T_cam_obj[t], ax), K)
            pose = np.where(o.valid[:, None, None], o.T_cam_obj, np.nan) if o.mesh_path else None
            objects.append({"object_id": o.object_id, "label": o.label, "valid": o.valid,
                            "bbox_2d": o.bbox_2d, "corners_2d": corners, "axes_2d": axes,
                            "dimensions": o.dimensions, "mesh_path": o.mesh_path,
                            "T_cam_obj": pose[:, :3] if pose is not None else None,
                            "depth": [float(o.T_cam_obj[t, 2, 3]) if o.valid[t] else None
                                      for t in range(len(fi))]})
        # Derived contact (HOI4D): signed hand-to-box distance per hand, object, frame; same
        # function and threshold as the `contact` tensors, so the viewer shows what training sees.
        contact = None
        if "contact" in meta.modalities and objects:
            d = hand_object_distance(ann)
            contact = {"dist": {side: d[s] for s, side in enumerate(HAND_SIDES)},
                       "threshold": DEFAULT_THRESHOLD_M}
        # Camera pose per frame as world->camera 3x4, to project the static scene cloud.
        T_cw = None
        if ann.T_world_cam is not None and np.isfinite(ann.T_world_cam).any():
            T_cw = np.full((len(fi), 3, 4), np.nan, np.float32)
            ok = np.isfinite(ann.T_world_cam).all(axis=(1, 2))
            T_cw[ok] = np.linalg.inv(ann.T_world_cam[ok])[:, :3]
        frames = {"start": start, "end": end, "hands": hands, "objects": objects, "contact": contact,
                  "T_cam_world": T_cw, "polygons": _polygon_chunk(dataset, sid, start, end),
                  "gaze_2d": ann.gaze_2d if ann.gaze_2d is not None and np.isfinite(ann.gaze_2d).any() else None}
        return _clean(frames)

    # ------------------------------------------------------------------ MANO hand mesh
    _mano: dict[str, Any] = {}

    def mano_model():
        if "m" not in _mano:
            from robot.hands.mano import ManoModel
            try:
                _mano["m"] = ManoModel()
            except Exception as e:  # smplx or the MANO pickles missing
                raise HTTPException(503, f"MANO model unavailable: {e}") from e
        return _mano["m"]

    @app.get("/api/hand_faces")
    def hand_faces() -> Any:
        m = mano_model()
        return {side: m.faces(side).ravel().tolist() for side in HAND_SIDES}

    @lru_cache(maxsize=16)
    def _hand_mesh(dataset: str, sid: str, start: int, end: int) -> bytes:
        ann = _raw_annotations(dataset, sid, start, end)
        T = len(ann.frame_indices)
        out = np.full((len(HAND_SIDES), T, MANO_VERTS, 3), -32768, np.int16)
        for s, side in enumerate(HAND_SIDES):
            h = ann.hands[side]
            v = h.valid & np.isfinite(h.mano_pose).all(1) & np.isfinite(h.mano_trans).all(1)
            if v.any():
                verts = mano_model().forward(side, h.mano_pose[v], np.nan_to_num(h.mano_betas[v]), h.mano_trans[v])[1]
                out[s, v] = np.clip(np.round(verts * 1e4), -32767, 32767)  # 0.1 mm units, camera frame
        return out.tobytes()

    @app.get("/api/sequence/{dataset}/{sid}/handmesh")
    def handmesh(dataset: str, sid: str, start: int = 0, end: int | None = None) -> Response:
        """MANO vertices for frames [start, end) as little-endian int16 (2, T, 778, 3) in 0.1 mm,
        camera frame; -32768 marks a missing hand. The client projects and shades them."""
        meta = st.meta(dataset, sid)
        if not meta.has_hand_pose:
            raise HTTPException(404, f"no hand pose for {dataset}/{sid}")
        end = min(end if end is not None else meta.num_frames, meta.num_frames, start + MAX_ANN_FRAMES)
        return Response(_hand_mesh(dataset, sid, max(start, 0), end), media_type="application/octet-stream",
                        headers={"Cache-Control": "max-age=3600"})

    # ------------------------------------------------------------------ static scene cloud
    @lru_cache(maxsize=8)
    def _scene(dataset: str, sid: str, n: int) -> bytes:
        meta = st.meta(dataset, sid)
        sc = st.loader.adapter(dataset).load_scene(meta) if "scene" in meta.modalities else None
        if sc is None or not len(sc.xyz):
            raise HTTPException(404, f"no scene cloud for {dataset}/{sid}")
        k = np.random.default_rng(0).permutation(len(sc.xyz))[:n]
        N = len(k)
        return (np.uint32(N).tobytes() + sc.xyz[k].astype("<f4").tobytes()
                + sc.label[k].astype("<i2").tobytes() + sc.rgb[k].astype(np.uint8).tobytes())

    @app.get("/api/scene/{dataset}/{sid}")
    def scene(dataset: str, sid: str, n: int = 40000) -> Response:
        """Random subset of the static scene cloud (world frame): uint32 N, then float32 xyz (N,3),
        int16 label (N,), uint8 rgb (N,3)."""
        return Response(_scene(dataset, sid, int(min(max(n, 1000), 200000))), media_type="application/octet-stream",
                        headers={"Cache-Control": "max-age=3600"})

    @lru_cache(maxsize=256)
    def _mesh(dataset: str, rel: str) -> dict[str, Any]:
        from robot.data.mesh import load_mesh

        root = st.loader.adapter(dataset).root.resolve()
        p = (root / rel).resolve()
        if p.suffix != ".obj" or not p.is_relative_to(root) or not p.is_file():
            raise HTTPException(404, f"no mesh {rel}")
        m = load_mesh(p, dataset, rel)
        return {"path": rel, "vertices": np.round(m.vertices, 4).ravel().tolist(),
                "faces": m.faces.ravel().tolist()}

    @app.get("/api/mesh/{dataset}")
    def mesh(dataset: str, path: str) -> Any:
        """Simplified mesh (object frame) of an object's ``mesh_path``, for the overlay."""
        if dataset not in ADAPTERS:
            raise HTTPException(404, f"unknown dataset {dataset}")
        return JSONResponse(_mesh(dataset, path))

    @app.get("/api/sequence/{dataset}/{sid}/annotations")
    def annotations(dataset: str, sid: str, start: int = 0, end: int | None = None) -> Any:
        meta = st.meta(dataset, sid)
        end = min(end if end is not None else meta.num_frames, meta.num_frames, start + MAX_ANN_FRAMES)
        return JSONResponse(_annotations(dataset, sid, max(start, 0), end))

    # ------------------------------------------------------------------ standard-format clip
    @app.get("/api/sequence/{dataset}/{sid}/clip")
    def clip(dataset: str, sid: str, start_s: float = 0.0, num: int = 8, fps: float | None = None) -> Any:
        """Run the real loader (without decoding frames) and describe the resulting ``Clip``."""
        meta = st.meta(dataset, sid)
        fi = frame_indices_at(meta, start_s, min(num, 64), fps)
        c = st.loader.load_clip(dataset, sid, fi, with_frames=False)
        a = c.annotations
        arrays, sidecar = a.to_arrays()
        return _clean({
            "frame_indices": a.frame_indices, "timestamps": a.timestamps,
            "K": c.K, "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                                     "finite": int(np.isfinite(v).sum()) if v.dtype.kind == "f" else None}
                                 for k, v in arrays.items()},
            "objects": sidecar["objects"], "actions": sidecar["actions"],
            "valid": {**{f"hand_{s}": int(a.hands[s].valid.sum()) for s in HAND_SIDES},
                      **{f"obj_{o.label}": int(o.valid.sum()) for o in a.objects},
                      "camera": int(np.isfinite(a.T_world_cam).all(axis=(1, 2)).sum())
                      if a.T_world_cam is not None else 0},
        })

    # ------------------------------------------------------------------ other modalities
    @app.get("/api/modality/{dataset}/{sid}/{kind}/{index}.png")
    def modality_image(dataset: str, sid: str, kind: str, index: int, w: int = 640) -> Response:
        """Depth (turbo colour map, 0.2-2.0 m) or instance segmentation (VOC colours) of one
        frame as a PNG with alpha, for overlaying on the video."""
        from PIL import Image

        from robot.data.adapters.hoi4d import voc_palette

        meta = st.meta(dataset, sid)
        if kind not in ("depth", "seg") or kind not in meta.modalities:
            raise HTTPException(404, f"{kind} not available for {dataset}/{sid}")
        ad = st.loader.adapter(dataset)
        i = np.array([min(max(index, 0), meta.num_frames - 1)])
        if kind == "depth":
            d = ad.load_depth(meta, i)[0]
            x = np.clip((d - 0.2) / 1.8, 0, 1)
            rgb = (_turbo(x) * 255).astype(np.uint8)
            alpha = np.where(d > 0, 255, 0).astype(np.uint8)
        else:
            ids = ad.load_seg(meta, i)[0]
            rgb = voc_palette()[ids]
            alpha = np.where(ids > 0, 255, 0).astype(np.uint8)
        img = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
        img.thumbnail((w, w), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=86400"})

    @lru_cache(maxsize=16)
    def _imu_summary(dataset: str, sid: str, points: int) -> dict:
        meta = st.meta(dataset, sid)
        imu = st.loader.adapter(dataset).load_imu(meta)
        if imu is None or len(imu.t) < 2:
            return {"t": [], "accel": [], "gyro": [], "frame": None}
        # Average into ``points`` bins so an hour of 200 Hz data stays a small JSON.
        edges = np.linspace(imu.t[0], imu.t[-1], points + 1)
        b = np.clip(np.searchsorted(edges, imu.t, side="right") - 1, 0, points - 1)
        n = np.bincount(b, minlength=points).astype(np.float64)
        n[n == 0] = np.nan
        mean = lambda x: np.stack([np.bincount(b, x[:, k], minlength=points) / n for k in range(3)], 1)  # noqa: E731
        return _clean({"t": (edges[:-1] + edges[1:]) / 2, "accel": mean(imu.accel), "gyro": mean(np.nan_to_num(imu.gyro)),
                       "frame": imu.frame})

    @app.get("/api/sequence/{dataset}/{sid}/imu")
    def imu(dataset: str, sid: str, points: int = 1500) -> Any:
        """IMU over the whole sequence, bin-averaged. The first call for an EPIC video parses
        its GoPro telemetry (seconds); later calls hit the cache."""
        meta = st.meta(dataset, sid)
        if "imu" not in meta.modalities:
            raise HTTPException(404, "no IMU for this sequence")
        return JSONResponse(_imu_summary(dataset, sid, min(max(points, 50), 5000)))

    # ------------------------------------------------------------------ media
    @app.get("/api/frame/{dataset}/{sid}/{index}.jpg")
    def frame(dataset: str, sid: str, index: int, w: int = 480) -> Response:
        """Decode one frame through ``VideoReader``. This is the path the loader takes, so the
        result shows what a model would actually receive."""
        from PIL import Image

        meta = st.meta(dataset, sid)
        with VideoReader(st.loader.adapter(dataset).video_path(meta)) as vr:
            img = Image.fromarray(vr.read_frame(min(max(index, 0), meta.num_frames - 1)))
        img.thumbnail((w, w))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return Response(buf.getvalue(), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})

    @app.get("/video/{dataset}/{sid}")
    def video(dataset: str, sid: str, request: Request) -> Response:
        meta = st.meta(dataset, sid)
        path = st.loader.adapter(dataset).video_path(meta)
        if not path.exists():
            raise HTTPException(404, "video missing")
        return _range_response(path, request.headers.get("range"))

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


def _turbo(x: np.ndarray) -> np.ndarray:
    """Google's Turbo colour map (polynomial approximation), x in [0, 1] -> RGB in [0, 1]."""
    x = np.clip(x, 0, 1)[..., None]
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return np.clip(np.concatenate([r, g, b], -1), 0, 1)


def _range_response(path: Path, range_header: str | None, chunk: int = 1 << 20) -> Response:
    """Serve a file with HTTP Range support, which the browser needs to seek in large videos."""
    size = path.stat().st_size
    ctype = mimetypes.guess_type(path.name)[0] or "video/mp4"
    start, end = 0, size - 1
    status = 200
    if range_header and range_header.startswith("bytes="):
        s, _, e = range_header[6:].split(",")[0].partition("-")
        if s:
            start = int(s)
            end = int(e) if e else size - 1
        else:  # suffix range: the last N bytes
            start = max(size - int(e), 0)
        end = min(end, size - 1)
        if start > end:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        status = 206

    def body():
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(end - start + 1)}
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(body(), status_code=status, media_type=ctype, headers=headers)
