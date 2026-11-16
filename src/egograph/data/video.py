"""Frame-accurate video access using PyAV.

``VideoReader.read(indices)`` returns exactly the frames with those 0-based indices,
counted in presentation order. Sorted index lists are decoded in a single forward pass
that seeks only when the next wanted frame is far ahead, so reading dense clips costs
about the same as linear decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    num_frames: int
    width: int
    height: int
    duration_s: float
    codec: str


def probe(path: str | Path) -> VideoInfo:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate or 30)
        if s.duration is not None and s.time_base is not None:
            duration = float(s.duration * s.time_base)
        elif c.duration is not None:
            duration = c.duration / 1e6
        else:
            duration = 0.0
        n = s.frames or int(round(duration * fps))
        return VideoInfo(fps, int(n), s.codec_context.width, s.codec_context.height, duration,
                         s.codec_context.name)


class VideoReader:
    # Seek instead of decoding through when the gap to the next wanted frame exceeds this
    # many frames (about two GOPs for typical egocentric footage).
    SEEK_GAP = 120

    def __init__(self, path: str | Path, threads: int = 0, pixel_format: str = "rgb24"):
        """``pixel_format`` is the numpy layout of decoded frames: "rgb24" gives (H, W, 3)
        uint8, and "gray16le" gives (H, W) uint16 (HOI4D depth videos)."""
        self.path = str(path)
        self.pixel_format = pixel_format
        self._c = av.open(self.path)
        self._s = self._c.streams.video[0]
        self._s.thread_type = "AUTO"
        if threads:
            self._s.codec_context.thread_count = threads
        self.fps = float(self._s.average_rate or self._s.guessed_rate or 30)
        self._tb = float(self._s.time_base)
        self._start = self._s.start_time or 0
        self.width = self._s.codec_context.width
        self.height = self._s.codec_context.height

    def close(self) -> None:
        self._c.close()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _frame_index(self, frame: av.VideoFrame) -> int:
        return int(round((frame.pts - self._start) * self._tb * self.fps))

    def _seek(self, index: int) -> None:
        # Land a little before the target. PyAV seeks to the preceding keyframe, and we
        # decode forward from there.
        ts = self._start + int(max(index - 1, 0) / self.fps / self._tb)
        self._c.seek(ts, stream=self._s, backward=True, any_frame=False)

    def read(self, indices: list[int] | np.ndarray) -> np.ndarray:
        """Return (N, H, W, 3) uint8 RGB frames in the order of ``indices``.

        If an index lies past the end of the stream, the last decoded frame is repeated.
        """
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            shape = (0, self.height, self.width) + (() if self.pixel_format.startswith("gray") else (3,))
            return np.zeros(shape, np.uint16 if "16" in self.pixel_format else np.uint8)
        wanted = np.unique(idx)
        out: dict[int, np.ndarray] = {}
        last: np.ndarray | None = None
        i = 0
        while i < len(wanted):
            self._seek(int(wanted[i]))
            eof = True
            tail = None  # last frame decoded, kept unconverted in case the stream ends early
            for frame in self._c.decode(self._s):
                fi = self._frame_index(frame)
                if fi < wanted[i]:
                    tail = frame
                    continue
                last = frame.to_ndarray(format=self.pixel_format)
                while i < len(wanted) and wanted[i] <= fi:
                    out[int(wanted[i])] = last
                    i += 1
                if i >= len(wanted) or wanted[i] - fi > self.SEEK_GAP:
                    eof = False
                    break  # done, or the next frame is far ahead, so seek again
            if eof:
                if tail is not None and (last is None or self._frame_index(tail) > fi):
                    last = tail.to_ndarray(format=self.pixel_format)
                if last is None:
                    raise RuntimeError(f"no frames decoded from {self.path} near frame {wanted[i]}")
                for w in wanted[i:]:
                    out[int(w)] = last
                break
        return np.stack([out[int(j)] for j in idx])

    def read_frame(self, index: int) -> np.ndarray:
        return self.read([index])[0]


def has_audio(path: str | Path) -> bool:
    with av.open(str(path)) as c:
        return len(c.streams.audio) > 0


def read_audio(path: str | Path, start_s: float, end_s: float, sample_rate: int = 16000,
               mono: bool = True) -> np.ndarray | None:
    """Decode the audio between ``start_s`` and ``end_s`` (video timeline) and resample it.

    Returns (S, C) float32 in [-1, 1] with S = round((end_s - start_s) * sample_rate), padded
    with zeros where the track is shorter, or None when the file has no audio stream.
    """
    n = max(int(round((end_s - start_s) * sample_rate)), 0)
    with av.open(str(path)) as c:
        if not c.streams.audio:
            return None
        s = c.streams.audio[0]
        layout = "mono" if mono else "stereo"
        res = av.AudioResampler(format="flt", layout=layout, rate=sample_rate)
        tb = float(s.time_base)
        c.seek(int(max(start_s - 0.5, 0) / tb), stream=s, backward=True, any_frame=False)
        chunks, t_first = [], None
        for frame in c.decode(s):
            if frame.pts is not None and frame.pts * tb > end_s + 0.1:
                break
            if t_first is None:
                t_first = (frame.pts or 0) * tb  # the resampler output is contiguous from here
            for rf in res.resample(frame):
                chunks.append(rf.to_ndarray().reshape(-1, 1 if mono else 2))
        ch = 1 if mono else 2
        out = np.zeros((n, ch), np.float32)
        if not chunks:
            return out
        pcm = np.concatenate(chunks)
        off = int(round((start_s - t_first) * sample_rate))
        src = pcm[max(off, 0):max(off, 0) + n - max(-off, 0)]
        out[max(-off, 0):max(-off, 0) + len(src)] = src
        return out
