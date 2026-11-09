"""Unified egocentric data pipeline: raw dataset -> adapter -> standard ``Clip``.

    from egograph.data import EgoLoader
    loader = EgoLoader()
    clip = loader.load_clip("hoi4d", seq_id, loader.frames_at(meta, start_s=0, num=16, fps=15))
"""

from egograph.data.loader import ClipWindow, EgoLoader, sample_windows
from egograph.data.schema import (ActionSegment, Annotations, Clip, HandTrack, ObjectTrack,
                                  SequenceMeta)

__all__ = ["ActionSegment", "Annotations", "Clip", "ClipWindow", "EgoLoader", "HandTrack",
           "ObjectTrack", "SequenceMeta", "sample_windows"]
