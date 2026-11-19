"""EPIC VISOR polygon masks: frame mapping, simplification, cache layout, viewer endpoints."""

from fractions import Fraction

import av
import numpy as np
import pytest

from conftest import require_index
from robot.data.adapters import visor
from robot.data.schema import CONTACT_NA, PolygonMasks


def _video(path, n, gap_after=None):
    """n frames at 60000/1001 fps (time base 1/60000), optionally with a one-frame timestamp gap."""
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=Fraction(60000, 1001))
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        s.time_base = Fraction(1, 60000)
        for k in range(n):
            f = av.VideoFrame.from_ndarray(np.full((48, 64, 3), k % 255, np.uint8), format="rgb24")
            f.pts = (k + (1 if gap_after is not None and k > gap_after else 0)) * 1001
            f.time_base = s.time_base
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def test_frame_map_matches_closed_form_without_gaps(tmp_path):
    _video(tmp_path / "v.mp4", 300)
    ns = np.arange(1, 290)
    got = visor.frame_map(tmp_path / "v.mp4", ns)
    want = -((-ns * 1000) // 1001) - 1          # ceil(n * 59.94 / 60) - 1
    assert (got == want).all()


def test_frame_map_follows_timestamps_across_a_gap(tmp_path):
    _video(tmp_path / "v.mp4", 300, gap_after=100)
    with av.open(str(tmp_path / "v.mp4")) as c:
        s = c.streams.video[0]
        pts = np.sort([p.pts for p in c.demux(s) if p.pts is not None])
        fps, start = float(s.average_rate), s.start_time or 0
    ns = np.arange(1, 290)
    got = visor.frame_map(tmp_path / "v.mp4", ns)
    t = (pts - start) / 60000
    for n, k in zip(ns, got):
        j = np.flatnonzero(t < n / 60)[-1]       # last frame shown before n/60 s
        assert k == round(t[j] * fps)


def test_simplify_and_window():
    sq = np.array([[0, 0], [5, 0], [10, 0], [10, 10], [0, 10]], float)
    assert visor.simplify(sq, 0.5).tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]
    a = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circ = np.stack([np.cos(a), np.sin(a)], 1) * 100
    for f in (visor.simplify, visor.simplify_np):
        c2 = f(circ, 0.75)
        assert 10 < len(c2) < 60
        r = np.hypot(*(c2 - 0).T)
        assert np.abs(r - 100).max() < 1e-3   # kept points lie on the circle (cv2 works in float32)
    assert visor.simplify_np(sq, 0.5).tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]
    pm = PolygonMasks(names=["right hand", "knife"], frames=np.array([5, 9]), manual=np.array([True, False]),
                      frame_start=np.array([0, 2, 3]), entity=np.array([0, 1, 1], np.int32),
                      contact=np.array([1, CONTACT_NA, CONTACT_NA], np.int32),
                      mask_start=np.array([0, 1, 3, 4]), poly_start=np.array([0, 3, 6, 9, 12]),
                      points=np.arange(24, dtype=np.int16).reshape(12, 2))
    w = pm.window(0, 9)
    assert [f["frame"] for f in w] == [5] and w[0]["manual"]
    assert [m["entity"] for m in w[0]["masks"]] == [0, 1] and w[0]["masks"][0]["contact"] == 1
    assert len(w[0]["masks"][1]["polygons"]) == 2 and w[0]["masks"][1]["polygons"][1][0].tolist() == [12, 13]
    assert [f["frame"] for f in pm.window(6, 100)] == [9]


@require_index("epic_kitchens")
def test_epic_visor_masks_and_viewer():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from robot.data import EgoLoader
    from robot.viewer.app import create_app

    L = EgoLoader()
    metas = [m for m in L.index("epic_kitchens").metas if "seg_sparse" in m.modalities]
    if not metas:
        pytest.skip("no VISOR masks downloaded")
    m = next((x for x in metas if x.sequence_id == "P01_03"), metas[0])
    pm = L.adapter("epic_kitchens").load_polygon_masks(m)
    assert (np.diff(pm.frames) > 0).all() and 0 <= pm.frames[0] and pm.frames[-1] < m.num_frames
    assert pm.manual.any() and (~pm.manual).any()
    hands = [i for i, n in enumerate(pm.names) if n.endswith("hand")]
    touch = pm.contact[np.isin(pm.entity, hands) & (pm.contact >= 0)]
    assert len(touch) and touch.max() < len(pm.names)
    assert pm.points.min() >= -50 and pm.points[:, 0].max() <= m.width + 50 and pm.points[:, 1].max() <= m.height + 50

    c = TestClient(create_app())
    base = f"/api/sequence/epic_kitchens/{m.sequence_id}"
    summ = c.get(base).json()["polygons"]
    assert summ["names"] == pm.names and summ["manual"] == pm.frames[pm.manual].tolist()
    f = int(pm.frames[pm.manual][0])
    start = f // 300 * 300
    ch = c.get(f"{base}/annotations?start={start}&end={start + 300}").json()["polygons"]
    e = next(x for x in ch if x[0] == f - start)
    assert e[1] is True and all(len(p) % 2 == 0 and len(p) >= 6 for mk in e[2] for p in mk[2:])
