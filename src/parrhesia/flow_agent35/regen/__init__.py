"""Regulation proposal generator ("regen") package."""

from .engine import propose_regulations_for_hotspot
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
]
