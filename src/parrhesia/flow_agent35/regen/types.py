"""Data models for the regulation generator module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple


__all__ = [
    "FlowScoreWeights",
    "RegenConfig",
    "FlowDiagnostics",
    "FlowScore",
    "Bundle",
    "Window",
    "RateCut",
    "BundleVariant",
    "PredictedImprovement",
    "Proposal",
]


@dataclass(frozen=True)
class FlowScoreWeights:
    """Weights for the linear flow scoring function."""

    w1: float = 10.0 # gH: xG / (xG + DH)
    w2: float = 10.0 # v_tilde
    w3: float = 0.0 # slack15
    w4: float = 0.0 # slack30
    w5: float = 0.25 # rho
    w6: float = 0.1 # coverage


@dataclass(frozen=True)
class RegenConfig:
    """Configuration controlling the regulation proposal generator."""

    g_min: float = 0.1
    rho_max: float = 0.7
    slack_min: float = 2.0
    window_margin_hours: float = 0.25
    min_window_hours: float = 1.0
    e_target_mode: str = "q95"
    fallback_to_peak: bool = True
    convert_occupancy_to_entrance: bool = False
    dwell_minutes: Optional[float] = None
    alpha_occupancy_to_entrance: float = 1.0
    local_search_steps: Tuple[int, ...] = None # (-2, -1, 1, 2)
    # When enabled, local search explores percentage adjustments around the target allowed rate
    # using deltas measured as a percentage of the original baseline rate r0_i.
    local_search_use_percent: bool = True
    local_search_percent_lower: float = 0.15  # 5%
    local_search_percent_upper: float = 0.85  # 50%
    local_search_percent_step: float = 0.1  # 5% increments
    max_variants_per_bundle: int = 64
    diversity_alpha: float = 0.2
    k_proposals: int = 6
    max_bundle_size: int = 6
    distinct_controls_required: bool = True
    autotrim_from_ctrl_to_hotspot: bool = False
    raise_on_edge_cases: bool = True


@dataclass(frozen=True)
class FlowDiagnostics:
    """Diagnostics summarising per-flow metrics used for scoring."""

    gH: float
    # gH_v_tilde: float
    v_tilde: float
    rho: float
    slack15: float
    slack30: float
    slack45: float
    coverage: float
    r0_i: float
    xGH: float
    DH: float
    tGl: int
    tGu: int
    bins_count: int
    num_flights: int


@dataclass(frozen=True)
class FlowScore:
    """Flow score and supporting diagnostics."""

    flow_id: int
    control_tv_id: Optional[str]
    score: float
    diagnostics: FlowDiagnostics
    num_flights: int


@dataclass(frozen=True)
class Bundle:
    """Collection of flows considered together for a regulation."""

    flows: List[FlowScore]
    weights_by_flow: Mapping[int, float]

    def flow_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(fs.flow_id for fs in self.flows))


@dataclass(frozen=True)
class Window:
    start_bin: int
    end_bin: int

    def duration(self) -> int:
        return int(self.end_bin - self.start_bin + 1)


@dataclass(frozen=True)
class RateCut:
    flow_id: int
    baseline_rate_r0: float
    cut_per_hour_lambda: int
    allowed_rate_R: int


@dataclass(frozen=True)
class BundleVariant:
    bundle: Bundle
    window: Window
    rates: Sequence[RateCut]


@dataclass(frozen=True)
class PredictedImprovement:
    delta_deficit_per_hour: float
    delta_objective_score: float


@dataclass(frozen=True)
class Proposal:
    hotspot_id: str
    controlled_volume: str
    window: Window
    flows_info: List[Mapping[str, Any]]
    predicted_improvement: PredictedImprovement
    diagnostics: Mapping[str, Any]
    # Regulation-level scope: target/ripple TVs and explicit cells
    target_cells: List[Tuple[str, int]]
    ripple_cells: List[Tuple[str, int]]
    target_tvs: List[str]
    ripple_tvs: List[str]
