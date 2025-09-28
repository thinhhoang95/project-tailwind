"""Configuration helpers for the regen module."""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .types import FlowScoreWeights, RegenConfig

__all__ = [
    "DEFAULT_WEIGHTS",
    "DEFAULT_CONFIG",
    "resolve_weights",
    "resolve_config",
]

DEFAULT_WEIGHTS = FlowScoreWeights()
DEFAULT_CONFIG = RegenConfig()


def resolve_weights(weights: Optional[FlowScoreWeights]) -> FlowScoreWeights:
    """Return the provided weights or the module defaults."""

    return weights if weights is not None else DEFAULT_WEIGHTS


def resolve_config(config: Optional[RegenConfig]) -> RegenConfig:
    """Return a defensive copy of the provided config or the defaults."""

    cfg = config if config is not None else DEFAULT_CONFIG
    # Return a shallow copy to avoid callers mutating shared defaults
    return replace(cfg)
