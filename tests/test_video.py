import av
import numpy as np

from egograph.data.video import VideoReader, probe


def _write(path, n=60, fps=30, gop=12):
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        s.codec_context.gop_size = gop
        s.options = {"bf": "2", "crf": "10"}
        for i in range(n):
            img = np.full((48, 64, 3), i * 4, np.uint8)  # frame i has brightness 4*i
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def test_random_access_is_frame_exact(tmp_path):
    p = tmp_path / "v.mp4"
    _write(p)
    info = probe(p)
    assert info.num_frames == 60 and abs(info.fps - 30) < 1e-6
    ids = [59, 0, 45, 46, 13, 13, 30, 1]
    with VideoReader(p) as vr:
        vr.SEEK_GAP = 5  # force the seek path as well as the decode-forward path
        frames = vr.read(ids)
    got = [int(round(f.mean() / 4)) for f in frames]
    assert got == ids
    with VideoReader(p) as vr:
        assert int(round(vr.read([200])[0].mean() / 4)) == 59  # past the end: last frame is repeated
