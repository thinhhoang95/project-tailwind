from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Sequence


@dataclass(frozen=True)
class Hotspot:
    tv_id: str
    bin: int


@dataclass(frozen=True)
class FlowSpec:
    flow_id: Hashable
    control_tv_id: Optional[str]
    flight_ids: Sequence[str]


@dataclass(frozen=True)
class RegulationProposal:
    flow_ids: List[Hashable]
    control_tv_id: str
    active_bins: List[int]
    rate_guess: int
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class HyperParams:
    # Matched-filter weights
    w_sum: float = 1.0
    w_max: float = 1.0
    kappa: float = 0.25
    alpha: float = 1.0
    beta: float = 1.0
    lambda_delay: float = 0.0
    # Eligibility
    q0: float = 1.0
    gamma: float = 5.0
    # Slack normalization
    S0: float = 1.0
    # Slack normalization mode: "x_at_argmin" (default), "x_at_control", or "constant"
    S0_mode: str = "x_at_argmin"
    # Price gap epsilon
    eps: float = 1e-6
    # Window sizes
    window_left: int = 2
    window_right: int = 2
    # Feature flag: use Rev1 pricing and score (α g_H ṽ − β ρ)
    use_rev1: bool = False
