"""Dataset viewer: a FastAPI backend over the same adapters and loader used for training.

Every overlay in the browser (hand keypoints, projected MANO joints, 3D object boxes,
camera trajectory) is computed by ``EgoLoader`` and ``egograph.data.geometry``. Looking at
the viewer therefore also checks the data pipeline.

Run on amdserver, where the data lives:   egograph viewer --host 0.0.0.0 --port 8765
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
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from egograph.data.adapters import ADAPTERS
from egograph.data.geometry import BOX_EDGES, box_corners, project, transform_points
from egograph.data.index import SequenceIndex, index_path
from egograph.data.loader import EgoLoader, frame_indices_at
from egograph.data.schema import HAND_PARENTS, HAND_SIDES
from egograph.data.video import VideoReader

STATIC = Path(__file__).parent / "static"
FACET_TAGS = ("participant", "split", "category_name", "human", "task", "part")
MAX_ANN_FRAMES = 900


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
        """Return the index, reloading it automatically after ``egograph index`` rewrites it."""
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
    app = FastAPI(title="egograph viewer")
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
                    facets={t: Counter(str(m.tags[t]) for m in ms if t in m.tags).most_common(60)
                            for t in FACET_TAGS if any(t in m.tags for m in ms[:50])},
                )
            out.append(info)
        return _clean(out)

    @app.get("/api/sequences")
    def sequences(dataset: str, q: str | None = None, require: list[str] = Query(default=[]),
                  tag: list[str] = Query(default=[]), offset: int = 0, limit: int = 100) -> Any:
        idx = st.index(dataset)
        if idx is None:
            return {"total": 0, "items": []}
        tags = dict(t.split(":", 1) for t in tag if ":" in t)
        ms = [m for m in idx.filter(query=q, require=require)
              if all(str(m.tags.get(k)) == v for k, v in tags.items())]
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
                       "hand_parents": HAND_PARENTS, "box_edges": BOX_EDGES})

    # ------------------------------------------------------------------ annotations
    @lru_cache(maxsize=64)
    def _annotations(dataset: str, sid: str, start: int, end: int) -> dict:
        meta = st.meta(dataset, sid)
        fi = np.arange(start, end)
        ann = st.loader.adapter(dataset).load_annotations(meta, fi)
        K = meta.K
        frames: dict[str, Any] = {}
        hands = {}
        for side in HAND_SIDES:
            h = ann.hands[side]
            if not h.valid.any():
                continue
            mano2d = project(h.joints_3d, K) if K is not None else None
            hands[side] = {"valid": h.valid, "joints_2d": h.joints_2d,
                           "mano_2d": mano2d if mano2d is not None and np.isfinite(mano2d).any() else None}
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
            objects.append({"object_id": o.object_id, "label": o.label, "valid": o.valid,
                            "bbox_2d": o.bbox_2d, "corners_2d": corners, "axes_2d": axes,
                            "dimensions": o.dimensions,
                            "depth": [float(o.T_cam_obj[t, 2, 3]) if o.valid[t] else None
                                      for t in range(len(fi))]})
        frames = {"start": start, "end": end, "hands": hands, "objects": objects}
        return _clean(frames)

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
