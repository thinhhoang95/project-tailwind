"""Regulation proposal generator ("regen") package."""

from .engine import propose_regulations_for_hotspot
from .hotspot_segment_extractor import (
    extract_hotspot_segments_from_resources,
    segment_to_hotspot_payload,
)
from .types import (
    Bundle,
    BundleVariant,
    FlowDiagnostics,
    FlowScore,
    FlowScoreWeights,
    PredictedImprovement,
    Proposal,
    RateCut,
    RegenConfig,
    Window,
)

__all__ = [
    "propose_regulations_for_hotspot",
    "Bundle",
    "BundleVariant",
    "FlowDiagnostics",
    "FlowScore",
    "FlowScoreWeights",
    "PredictedImprovement",
    "Proposal",
    "RateCut",
    "RegenConfig",
    "Window",
    "extract_hotspot_segments_from_resources",
    "segment_to_hotspot_payload",
]
