"""Interface for 3D hand and object pose extractors (Month 2 baselines: StableHand, EgoGrasp,
WHOLE, plus HaMeR- or WiLoR-style per-frame methods).

An extractor takes a standard ``Clip`` and returns ``Annotations`` in the same schema that
HOI4D ground truth uses. That makes predictions and GT directly comparable, and lets the
viewer and the graph builder in Month 3 consume either one.

    class StableHandExtractor(PoseExtractor):
        name = "stablehand"
        def extract(self, clip): ...

Predictions are cached per window under ``runs/<run>/predictions/<dataset>/<seq>/<start>/``
with ``Clip.save``, so evaluation never re-runs inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from egograph.data.schema import Annotations, Clip

EXTRACTORS: dict[str, type["PoseExtractor"]] = {}


def register_extractor(cls: type["PoseExtractor"]) -> type["PoseExtractor"]:
    EXTRACTORS[cls.name] = cls
    return cls


class PoseExtractor(ABC):
    name: ClassVar[str]
    #: What the method needs: frames are always given; K may be None (uncalibrated video).
    needs_intrinsics: ClassVar[bool] = False

    @abstractmethod
    def extract(self, clip: Clip) -> Annotations:
        """Predict hands, objects, and/or the camera trajectory for every frame of ``clip``."""


@register_extractor
class GroundTruthExtractor(PoseExtractor):
    """Passes the dataset annotations through unchanged. Useful as an upper bound and to test
    the evaluation and visualisation code paths before any baseline is set up."""

    name = "ground_truth"

    def extract(self, clip: Clip) -> Annotations:
        return clip.annotations
