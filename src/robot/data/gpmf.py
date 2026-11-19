"""GoPro GPMF telemetry: accelerometer and gyroscope from the ``gpmd`` track of an MP4.

Most EPIC-KITCHENS-100 videos (GoPro Hero5/Hero7) keep this track. GPMF is a KLV stream:
each item is a 4-char key, a 1-byte type, a 1-byte struct size, a 2-byte big-endian
repeat count, and a payload padded to 4 bytes. Type 0 means "nested". ACCL and GYRO are
int16 triples divided by the preceding SCAL. ``ORIN`` names the camera axis of each stored
channel (lowercase = negated), for example "YxZ" (Hero7) or "ZXY" (Hero5).

Axis convention (checked on 10 EPIC videos, 2026-11-16): after reordering by ORIN, the
mean accelerometer reading is about (0, +8.3, +4.8) m/s^2 for head-mounted footage, so the
GoPro axes are x right, y up, z backward (out of the back of the camera), and up has a
negative component along the viewing direction. That matches a ~30 deg downward head
tilt. ``to_opencv`` maps them to the OpenCV camera frame used everywhere else: (x, -y, -z).

Each gpmd packet covers ~1 s (its pts and duration). Samples inside a packet are spread
evenly over that span, as the GPMF parser reference implementation does.
"""

from __future__ import annotations

import struct
from pathlib import Path

import av
import numpy as np

GOPRO_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def _walk(buf: bytes, cb) -> None:
    i = 0
    while i + 8 <= len(buf):
        key = buf[i:i + 4]
        typ = buf[i + 4]
        size = buf[i + 5]
        rep = struct.unpack(">H", buf[i + 6:i + 8])[0]
        n = size * rep
        payload = buf[i + 8:i + 8 + n]
        if typ == 0:
            _walk(payload, cb)
        else:
            cb(key, chr(typ), size, rep, payload)
        i += 8 + ((n + 3) // 4) * 4


def _scale(payload: bytes, typ: str) -> float:
    fmt = {"s": ">h", "S": ">H", "l": ">i", "L": ">I", "f": ">f"}.get(typ, ">h")
    return float(struct.unpack(fmt, payload[:struct.calcsize(fmt)])[0])


def _orin_to_xyz(raw: np.ndarray, orin: str) -> np.ndarray:
    out = np.empty_like(raw)
    for ch, letter in enumerate(orin[:3]):
        out[:, "XYZ".index(letter.upper())] = raw[:, ch] * (1.0 if letter.isupper() else -1.0)
    return out


def telemetry_stream(container: av.container.InputContainer):
    for s in container.streams:
        if s.type == "data" and "MET" in (s.metadata.get("handler_name") or ""):
            return s
    return None


def has_telemetry(path: str | Path) -> bool:
    with av.open(str(path)) as c:
        return telemetry_stream(c) is not None


def read_imu(path: str | Path) -> dict[str, np.ndarray] | None:
    """Parse the whole telemetry track. Returns {"t", "accel", "gyro"} in the OpenCV camera
    frame (t in seconds on the video timeline; accel and gyro may have different lengths, so
    gyro is interpolated onto the accelerometer times), or None without a gpmd track.

    This reads the entire file (the track is interleaved with the video), so callers cache it.
    """
    with av.open(str(path)) as c:
        s = telemetry_stream(c)
        if s is None:
            return None
        tb = float(s.time_base)
        series: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"ACCL": [], "GYRO": []}
        for pkt in c.demux(s):
            if not pkt.size or pkt.pts is None:
                continue
            t0 = pkt.pts * tb
            dur = (pkt.duration or 0) * tb or 1.0
            cur: dict[str, object] = {}

            def cb(key, typ, size, rep, payload):
                k = key.decode("latin1")
                if k == "STRM":
                    return
                if k == "SCAL":
                    cur["scal"] = _scale(payload, typ)
                elif k == "ORIN":
                    cur["orin"] = payload.decode("latin1")
                elif k in series and typ == "s":
                    raw = np.frombuffer(payload, ">i2").reshape(rep, size // 2).astype(np.float32)
                    raw = raw[:, :3] / float(cur.get("scal", 1.0) or 1.0)
                    # Hero5 omits ORIN; its documented channel order is Z, X, Y.
                    xyz = _orin_to_xyz(raw, str(cur.get("orin", "ZXY")))
                    ts = t0 + (np.arange(rep) + 0.5) * dur / rep
                    series[k].append((ts, xyz @ GOPRO_TO_OPENCV))
                    cur.pop("orin", None)
                    cur.pop("scal", None)

            _walk(bytes(pkt), cb)
    if not series["ACCL"]:
        return None
    t = np.concatenate([a for a, _ in series["ACCL"]])
    accel = np.concatenate([b for _, b in series["ACCL"]])
    gyro = np.full_like(accel, np.nan)
    if series["GYRO"]:
        gt = np.concatenate([a for a, _ in series["GYRO"]])
        gv = np.concatenate([b for _, b in series["GYRO"]])
        for k in range(3):
            gyro[:, k] = np.interp(t, gt, gv[:, k])
    order = np.argsort(t, kind="stable")
    return {"t": t[order], "accel": accel[order].astype(np.float32), "gyro": gyro[order].astype(np.float32)}
