from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np

RegulationMode = Literal["per_flow", "blanket"]


@dataclass(frozen=True)
class RegulationSpec:
    """Immutable description of a committed regulation in the flow plan."""

    control_volume_id: str
    window_bins: Tuple[int, int]
    flow_ids: Tuple[str, ...]
    mode: RegulationMode = "per_flow"
    committed_rates: Optional[Union[int, Dict[str, int]]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t0, t1 = self.window_bins
        if t0 >= t1:
            raise ValueError("window_bins must define a non-empty half-open interval")
        if any(b < 0 for b in self.window_bins):
            raise ValueError("window_bins must be non-negative")
        object.__setattr__(self, "flow_ids", tuple(sorted(str(fid) for fid in self.flow_ids)))

    def to_canonical_dict(self) -> Dict[str, Any]:
        rates = self.committed_rates
        if isinstance(rates, dict):
            rates = {k: int(v) for k, v in sorted(rates.items())}
        elif rates is not None:
            rates = int(rates)
        return {
            "control_volume_id": self.control_volume_id,
            "window_bins": list(self.window_bins),
            "flow_ids": list(self.flow_ids),
            "mode": self.mode,
            "committed_rates": rates,
        }

    def with_committed_rates(self, rates: Union[int, Dict[str, int]]) -> "RegulationSpec":
        """Return a copy with committed rates populated."""
        return replace(self, committed_rates=rates)


@dataclass(frozen=True)
class HotspotContext:
    """Mutable planning context while constructing a regulation."""

    control_volume_id: str
    window_bins: Tuple[int, int]
    candidate_flow_ids: Tuple[str, ...]
    mode: RegulationMode = "per_flow"
    selected_flow_ids: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t0, t1 = self.window_bins
        if t0 >= t1:
            raise ValueError("window_bins must define a non-empty interval")
        if any(b < 0 for b in self.window_bins):
            raise ValueError("window_bins must be non-negative")
        object.__setattr__(
            self,
            "candidate_flow_ids",
            tuple(sorted(str(fid) for fid in self.candidate_flow_ids)),
        )
        object.__setattr__(
            self,
            "selected_flow_ids",
            tuple(sorted(str(fid) for fid in self.selected_flow_ids)),
        )

    def add_flow(self, flow_id: str) -> "HotspotContext":
        fid = str(flow_id)
        if fid not in self.candidate_flow_ids:
            raise KeyError(f"flow {fid} not in candidate set")
        if fid in self.selected_flow_ids:
            return self
        return replace(
            self,
            selected_flow_ids=tuple(sorted(self.selected_flow_ids + (fid,))),
        )

    def remove_flow(self, flow_id: str) -> "HotspotContext":
        fid = str(flow_id)
        if fid not in self.selected_flow_ids:
            return self
        remaining = tuple(x for x in self.selected_flow_ids if x != fid)
        return replace(self, selected_flow_ids=remaining)

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "control_volume_id": self.control_volume_id,
            "window_bins": list(self.window_bins),
            "candidate_flow_ids": list(self.candidate_flow_ids),
            "selected_flow_ids": list(self.selected_flow_ids),
            "mode": self.mode,
        }


StageLiteral = Literal["idle", "select_hotspot", "select_flows", "confirm", "stopped"]


@dataclass
class PlanState:
    """Snapshot of the planning process prior to MCTS."""

    plan: List[RegulationSpec] = field(default_factory=list)
    hotspot_context: Optional[HotspotContext] = None
    z_hat: Optional[np.ndarray] = None
    stage: StageLiteral = "idle"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "PlanState":
        clone = PlanState(
            plan=list(self.plan),
            hotspot_context=self.hotspot_context,
            z_hat=None if self.z_hat is None else np.array(self.z_hat, copy=True),
            stage=self.stage,
            metadata=dict(self.metadata),
        )
        return clone

    def canonical_key(self) -> str:
        """Return a stable JSON key for caching purposes."""
        payload = {
            "plan": [reg.to_canonical_dict() for reg in self.plan],
            "hotspot_context": self.hotspot_context.to_canonical_dict()
            if self.hotspot_context
            else None,
            "stage": self.stage,
            "z_hat": self._z_hat_signature(),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _z_hat_signature(self) -> Optional[List[float]]:
        if self.z_hat is None:
            return None
        return [float(x) for x in np.asarray(self.z_hat, dtype=float).tolist()]

    def reset_hotspot(self, *, next_stage: Optional[StageLiteral] = None) -> None:
        self.hotspot_context = None
        self.z_hat = None
        self.metadata.pop("awaiting_commit", None)
        self.stage = next_stage or "idle"

    def set_hotspot(self, context: HotspotContext, *, z_hat_shape: Optional[int] = None) -> None:
        self.hotspot_context = context
        self.stage = "select_flows"
        if z_hat_shape is None or z_hat_shape <= 0:
            self.z_hat = None
        else:
            self.z_hat = np.zeros(z_hat_shape, dtype=float)

    def ensure_stage(self, expected: Iterable[StageLiteral]) -> None:
        if self.stage not in set(expected):
            raise RuntimeError(f"state stage {self.stage!r} not in {list(expected)!r}")


__all__ = ["PlanState", "RegulationSpec", "HotspotContext", "RegulationMode", "StageLiteral"]
