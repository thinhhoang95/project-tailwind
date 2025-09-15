from .types import Hotspot, FlowSpec, RegulationProposal, HyperParams
from .base_caches import build_base_caches, attention_mask_from_cells
from .travel_offsets import minutes_to_bin_offsets, flow_offsets_from_ctrl
from .flow_signals import build_flow_g0, build_xG_series
from .per_flow_features import (
    phase_time,
    price_kernel_vG,
    price_to_hotspot_vGH,
    price_kernel_vG_legacy,
    price_to_hotspot_vGH_legacy,
    slack_G_at,
    eligibility_a,
    slack_penalty,
    score,
    score_legacy,
    # Rev1
    mass_weight_gH,
    price_contrib_v_tilde,
    score_rev1,
)
from .pairwise_features import (
    temporal_overlap,
    offset_orthogonality,
    slack_profile,
    slack_corr,
    price_gap,
)

__all__ = [
    # Types
    "Hotspot",
    "FlowSpec",
    "RegulationProposal",
    "HyperParams",
    # Caches / helpers
    "build_base_caches",
    "attention_mask_from_cells",
    "minutes_to_bin_offsets",
    "flow_offsets_from_ctrl",
    "build_flow_g0",
    "build_xG_series",
    # Per-flow features
    "phase_time",
    "price_kernel_vG",
    "price_to_hotspot_vGH",
    "price_kernel_vG_legacy",
    "price_to_hotspot_vGH_legacy",
    "slack_G_at",
    "eligibility_a",
    "slack_penalty",
    "score",
    "score_legacy",
    # Rev1
    "mass_weight_gH",
    "price_contrib_v_tilde",
    "score_rev1",
    # Pairwise features
    "temporal_overlap",
    "offset_orthogonality",
    "slack_profile",
    "slack_corr",
    "price_gap",
]
