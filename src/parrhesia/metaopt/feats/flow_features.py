from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np

from ..types import Hotspot, HyperParams
from ..base_caches import build_base_caches
from ..travel_offsets import minutes_to_bin_offsets, flow_offsets_from_ctrl
from ..per_flow_features import (
    phase_time,
    mass_weight_gH,
    price_contrib_v_tilde,
    slack_G_at,
    slack_penalty,
)


@dataclass(frozen=True)
class FlowFeatures:
    flow_id: int
    control_tv_id: Optional[str]

    # Phase time bounds (control volume time across hotspot period)
    tGl: int
    tGu: int

    # Aggregated (sum over bins in the hotspot period)
    xGH: float                 # sum of x̂_GH over period
    DH: float                  # sum of D_H over period
    gH_sum: float              # sum of per-bin g_H over period
    gH_avg: float              # average of per-bin g_H over period (over bins where t_G is valid)
    gH: float                  # derived at period level: xGH / (xGH + DH)
    v_tilde: float             # sum of per-bin ṽ over period
    #  gH_v_tilde: float          # derived at period level: gH (derived) * v_tilde (sum)

    # Slack sums and global-argmin row indices per Δ ∈ {0, 15, 30, 45} minutes
    Slack_G0: float
    Slack_G0_row: Optional[int]
    Slack_G0_occ: Optional[float]
    Slack_G0_cap: Optional[float]
    Slack_G15: float
    Slack_G15_row: Optional[int]
    Slack_G15_occ: Optional[float]
    Slack_G15_cap: Optional[float]
    Slack_G30: float
    Slack_G30_row: Optional[int]
    Slack_G30_occ: Optional[float]
    Slack_G30_cap: Optional[float]
    Slack_G45: float
    Slack_G45_row: Optional[int]
    Slack_G45_occ: Optional[float]
    Slack_G45_cap: Optional[float]

    # Risk penalty (sum over period)
    rho: float

    # Auxiliary counts
    bins_count: int
    num_flights: int


class FlowFeaturesExtractor:
    """
    Compute per-flow, non-pairwise features over a hotspot time period.

    Inputs
    - indexer, flight_list, travel_minutes_map, capacities_by_tv are expected to be
      consistent with the environment used to build flows and caches.
    - flows_payload (optional) should be the output of parrhesia.api.flows.compute_flows.

    Outputs
    - Dict[flow_id, FlowFeatures] aggregating per-bin contributions across the period.
    - Implements both gH sum and average variants; also exposes derived gH = xGH/(xGH+DH).
    - Slack row indices are single global argmin per Δ across the period.
    - S0_mode default remains "x_at_argmin".
    """

    def __init__(
        self,
        indexer: Any,
        flight_list: Any,
        capacities_by_tv: Mapping[str, np.ndarray],
        travel_minutes_map: Mapping[str, Mapping[str, float]],
        params: Optional[HyperParams] = None,
        *,
        autotrim_from_ctrl_to_hotspot: bool = False,
    ) -> None:
        self.indexer = indexer
        self.flight_list = flight_list
        self.capacities_by_tv = capacities_by_tv
        self.travel_minutes_map = travel_minutes_map
        self.params = params or HyperParams()
        # Behavior flag: if True, trim flight sequences and τ rows to the
        # earliest hotspot visit; otherwise use the full set of TVs that any
        # flight in the flow reaches (default = False as requested).
        self.autotrim_from_ctrl_to_hotspot: bool = bool(autotrim_from_ctrl_to_hotspot)

        # Build caches once
        self.caches: Dict[str, Any] = build_base_caches(
            flight_list=self.flight_list,
            capacities_by_tv=self.capacities_by_tv,
            indexer=self.indexer,
        )
        self.H_bool: np.ndarray = self.caches["hourly_excess_bool"]
        self.S_mat: np.ndarray = self.caches["slack_per_bin_matrix"]
        self.rolling_occ_by_bin: Optional[np.ndarray] = self.caches.get("rolling_occ_by_bin")
        self.hourly_capacity_matrix: Optional[np.ndarray] = self.caches.get("hourly_capacity_matrix")
        self.bins_per_hour: Optional[int] = int(self.caches.get(
            "bins_per_hour", max(1, 60 // int(self.indexer.time_bin_minutes))
        ))

        # Precompute travel bin offsets
        self.bin_offsets: Mapping[str, Mapping[str, int]] = minutes_to_bin_offsets(
            self.travel_minutes_map, time_bin_minutes=int(self.indexer.time_bin_minutes)
        )

        # Row index mapping utilities
        self.row_map: Mapping[str, int] = self.flight_list.tv_id_to_idx
        self.idx_to_tv_id: Mapping[int, str] = {idx: tv for tv, idx in self.row_map.items()}

        self.T: int = int(self.indexer.num_time_bins)

    def _build_xG_series_from_flows(
        self,
        flows_payload: Mapping[str, Any],
        *,
        hotspot_tv: Optional[str] = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]], Dict[int, Optional[str]]]:
        """
        Rebuild xG and τ maps from compute_flows payload; return (xG_map, tau_map, ctrl_by_flow).

        Behavior differences vs. earlier revision:
        - If a hotspot is provided, infer τ with trim_policy="earliest_hotspot" and pass
          that hotspot to the offsets helper so sign inference is restricted to the
          prefix up to the first hotspot visit per flight.
        - Restrict τ rows to TVs actually visited by the flow in that same prefix and
          always include the hotspot row to preserve the primary term alignment.
        """
        T = self.T
        xG_map: Dict[int, np.ndarray] = {}
        tau_map: Dict[int, Dict[int, int]] = {}
        ctrl_by_flow: Dict[int, Optional[str]] = {}
        flights_by_flow: Dict[int, List[str]] = {}

        # Assume hotspot tv exists in row_map if called correctly
        for fobj in flows_payload.get("flows", []):
            fid = int(fobj.get("flow_id"))
            ctrl = fobj.get("controlled_volume")
            ctrl_by_flow[fid] = str(ctrl) if ctrl else None

            flights_specs = fobj.get("flights", [])
            flow_flight_ids: List[str] = [
                str(sp.get("flight_id")) for sp in flights_specs if sp.get("flight_id")
            ]
            flights_by_flow[int(fid)] = list(flow_flight_ids)
            # Demand series by requested bins
            x = np.zeros(T, dtype=float)
            for sp in flights_specs:
                rb = int(sp.get("requested_bin", 0))
                if 0 <= rb < T:
                    x[rb] += 1.0
            xG_map[fid] = x

            # Signed τ map from control to all TVs; restrict to visited rows in the
            # prefix up to the earliest hotspot visit, and always include the hotspot row.
            h_row: Optional[int] = None
            if hotspot_tv is not None and hotspot_tv in self.row_map:
                h_row = int(self.row_map[str(hotspot_tv)])

            tau = flow_offsets_from_ctrl(
                ctrl,
                self.row_map,
                self.bin_offsets,
                flow_flight_ids=flow_flight_ids,
                flight_list=self.flight_list,
                hotspots=([str(hotspot_tv)] if (self.autotrim_from_ctrl_to_hotspot and hotspot_tv is not None) else None),
                trim_policy=("earliest_hotspot" if self.autotrim_from_ctrl_to_hotspot else None),
                direction_sign_mode="order_vs_ctrl",
                tv_centroids=getattr(self.indexer, "tv_centroids", None) or getattr(self, "tv_centroids", None),
            ) or {}

            # Limit τ to rows the flow can touch: union of all TVs visited by at least
            # one flight in the flow. If autotrim is enabled, restrict sequences to
            # the prefix up to the earliest hotspot visit; otherwise keep full sequences.
            # Always include the hotspot row for stability.
            visited_rows: set[int] = set()
            for fid_str in flow_flight_ids:
                try:
                    seq = self.flight_list.get_flight_tv_sequence_indices(str(fid_str))
                    if not isinstance(seq, np.ndarray):
                        seq = np.asarray(seq, dtype=np.int64)
                except Exception:
                    continue
                if seq.size == 0:
                    continue
                # Optional trim to earliest hotspot, mirroring offsets' trim
                if self.autotrim_from_ctrl_to_hotspot and (h_row is not None):
                    cut: Optional[int] = None
                    for i, v in enumerate(seq.tolist()):
                        if int(v) == int(h_row):
                            cut = i
                            break
                    if cut is not None:
                        seq = seq[: cut + 1]
                visited_rows.update(int(v) for v in seq.tolist())

            if h_row is not None:
                visited_rows.add(int(h_row))

            tau_map[fid] = {int(r): int(off) for r, off in (tau or {}).items() if int(r) in visited_rows}

        # Stash for internal fallback use
        try:
            self._flights_by_flow = flights_by_flow
        except Exception:
            pass
        return xG_map, tau_map, ctrl_by_flow

    def _slack_min_row(
        self,
        t_val: int,
        tau_row_to_bins: Mapping[int, int],
    ) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float]]:
        """Return (row_index, min_slack_value, rolling_occ, capacity) at time t_val + τ; None if undefined."""
        V, T = self.S_mat.shape
        tau = np.zeros(int(V), dtype=np.int32)
        touched = np.zeros(int(V), dtype=np.bool_)
        for r, off in tau_row_to_bins.items():
            r_int = int(r)
            if 0 <= r_int < int(V):
                tau[r_int] = int(off)
                touched[r_int] = True
        if not np.any(touched):
            return None, None, None, None
        t_idx_vec_all = np.clip(int(t_val) + tau, 0, int(T) - 1)
        rows_all = np.arange(int(V), dtype=np.int32)
        rows = rows_all[touched]
        t_idx_vec = t_idx_vec_all[touched]

        roll_vals: Optional[np.ndarray] = None
        cap_vals: Optional[np.ndarray] = None
        try:
            if (
                self.rolling_occ_by_bin is not None
                and self.hourly_capacity_matrix is not None
                and self.bins_per_hour is not None
            ):
                max_hour_idx = int(self.hourly_capacity_matrix.shape[1]) - 1
                hour_idx = np.clip(t_idx_vec // int(self.bins_per_hour), 0, max_hour_idx)
                roll_vals = self.rolling_occ_by_bin[rows, t_idx_vec]
                cap_vals = self.hourly_capacity_matrix[rows, hour_idx]
                vals = cap_vals - roll_vals
            else:
                vals = self.S_mat[rows, t_idx_vec]
        except Exception:
            vals = self.S_mat[rows, t_idx_vec]
            roll_vals = None
            cap_vals = None

        if vals.size == 0:
            return None, None, None, None
        local_idx = int(np.argmin(vals))
        r_hat = int(rows[int(local_idx)])
        s_min = float(vals[int(local_idx)])
        occ_val: Optional[float] = None
        cap_val: Optional[float] = None
        if roll_vals is not None:
            occ_val = float(roll_vals[int(local_idx)])
        if cap_vals is not None:
            cap_val = float(cap_vals[int(local_idx)])
        return r_hat, s_min, occ_val, cap_val

    def compute_for_hotspot(
        self,
        hotspot_tv: str,
        timebins: Sequence[int],
        *,
        flows_payload: Optional[Mapping[str, Any]] = None,
        direction_opts: Optional[Mapping[str, Any]] = None,
    ) -> Dict[int, FlowFeatures]:
        """
        Compute per-flow features aggregated over the provided hotspot bins.

        If flows_payload is not provided, attempts to call parrhesia.api.flows.compute_flows
        using the given direction_opts.
        """
        # Guard inputs
        if not timebins:
            return {}
        if hotspot_tv not in self.row_map:
            raise ValueError(f"Unknown hotspot tv_id: {hotspot_tv}")

        # If caller provided tv_centroids in direction_opts, make them available for τ sign fallback
        if direction_opts and isinstance(direction_opts, Mapping):
            try:
                tvc = direction_opts.get("tv_centroids")
                if tvc is not None:
                    # Stash on self so _build_xG_series_from_flows can pick it up
                    setattr(self, "tv_centroids", tvc)
            except Exception:
                pass

        # Fetch or reuse flows
        if flows_payload is None:
            from parrhesia.api.flows import compute_flows  # lazy import to avoid hard dep at import time
            flows_payload = compute_flows(
                tvs=[hotspot_tv],
                timebins=list(timebins),
                direction_opts=dict(direction_opts or {}),
            )

        # Build per-flow demand and offsets respecting autotrim flag and keeping hotspot row
        xG_map, tau_map, ctrl_by_flow = self._build_xG_series_from_flows(
            flows_payload, hotspot_tv=str(hotspot_tv)
        )

        h_row = int(self.row_map[hotspot_tv])
        T = self.T
        minutes_per_bin = int(self.indexer.time_bin_minutes)
        # Map Δ minutes → bin shifts (robust to non-15-min bins)
        def _shift_bins(mins: int) -> int:
            if mins == 0:
                return 0
            # Choose nearest integer shift; ensure at least 1 if mins > 0
            return max(1, int(round(mins / float(minutes_per_bin))))

        delta_min_list = [0, 15, 30, 45]
        delta_to_shift = {m: _shift_bins(m) for m in delta_min_list}

        # Initialize aggregations
        accs: Dict[int, Dict[str, float]] = {}
        bins_count_by_flow: Dict[int, int] = {int(fid): 0 for fid in xG_map.keys()}
        tGl_by_flow: Dict[int, int] = {int(fid): T - 1 for fid in xG_map.keys()}
        tGu_by_flow: Dict[int, int] = {int(fid): 0 for fid in xG_map.keys()}

        # Slack sums and global argmin row trackers per Δ ∈ {0, 15, 30, 45} minutes (for example).
        slack_sums: Dict[int, Dict[int, float]] = {int(fid): {m: 0.0 for m in delta_min_list} for fid in xG_map.keys()}
        slack_min_rows: Dict[int, Dict[int, Optional[int]]] = {int(fid): {m: None for m in delta_min_list} for fid in xG_map.keys()}
        slack_min_vals: Dict[int, Dict[int, float]] = {int(fid): {m: float("inf") for m in delta_min_list} for fid in xG_map.keys()}
        slack_min_occ: Dict[int, Dict[int, Optional[float]]] = {int(fid): {m: None for m in delta_min_list} for fid in xG_map.keys()}
        slack_min_cap: Dict[int, Dict[int, Optional[float]]] = {int(fid): {m: None for m in delta_min_list} for fid in xG_map.keys()}

        # Initialize metric accumulators
        for fid in xG_map.keys():
            accs[int(fid)] = {
                "xGH": 0.0,
                "DH": 0.0,
                "gH_sum": 0.0,
                "v_tilde": 0.0,
                # "gH_v_tilde_sum": 0.0,  # per-bin product sum (optional)
                "rho": 0.0,
            }

        # Per-bin loop
        for b in timebins:
            hot = Hotspot(tv_id=hotspot_tv, bin=int(b))
            for fid in xG_map.keys():
                tau = dict(tau_map[int(fid)] or {})
                # Keep hotspot row to preserve primary term (if absent). If τ for hotspot
                # is somehow missing, fall back to the absolute offsets map rather than 0.
                if h_row not in tau:
                    try:
                        # Recompute τ with flow context to preserve signs; respect autotrim flag
                        ctrl_tv = ctrl_by_flow.get(int(fid))
                        flow_flight_ids = None
                        if hasattr(self, "_flights_by_flow"):
                            flow_flight_ids = getattr(self, "_flights_by_flow").get(int(fid))
                        # Prefer geometric signs when flight ids are unavailable but tv_centroids exist
                        tvc = getattr(self.indexer, "tv_centroids", None) or getattr(self, "tv_centroids", None)
                        if not flow_flight_ids:
                            if tvc:
                                warnings.warn(
                                    "[FlowFeaturesExtractor] Missing flow_flight_ids for τ recompute; "
                                    "using vector_centroid sign inference (geometric fallback). Results may differ from order-based signs.",
                                    RuntimeWarning,
                                )
                                dir_mode = "vector_centroid"
                            else:
                                warnings.warn(
                                    "[FlowFeaturesExtractor] Missing flow_flight_ids and tv_centroids for τ recompute; "
                                    "falling back to non-signed τ magnitudes. Derived times may be biased.",
                                    RuntimeWarning,
                                )
                                # Explicitly disable sign mode to force magnitude fallback
                                dir_mode = "disabled"
                        else:
                            # Normal case: order_vs_ctrl is the intended sign mode
                            dir_mode = "order_vs_ctrl"
                        tau_full = flow_offsets_from_ctrl(
                            ctrl_tv,
                            self.row_map,
                            self.bin_offsets,
                            flow_flight_ids=flow_flight_ids,
                            flight_list=self.flight_list if flow_flight_ids else None,
                            hotspots=([str(hotspot_tv)] if (self.autotrim_from_ctrl_to_hotspot and hotspot_tv is not None) else None),
                            trim_policy=("earliest_hotspot" if self.autotrim_from_ctrl_to_hotspot else None),
                            direction_sign_mode=dir_mode,
                            tv_centroids=tvc,
                        ) or {}
                        if int(h_row) in tau_full:
                            tau[int(h_row)] = int(tau_full[int(h_row)])
                        else:
                            warnings.warn(
                                "[FlowFeaturesExtractor] τ for hotspot row missing after recompute; defaulting to 0.",
                                RuntimeWarning,
                            )
                            tau[int(h_row)] = 0
                    except Exception:
                        warnings.warn(
                            "[FlowFeaturesExtractor] Exception during τ recompute for hotspot; defaulting τ(hotspot)=0.",
                            RuntimeWarning,
                        )
                        tau[int(h_row)] = 0

                xG = xG_map[int(fid)]
                tG = int(phase_time(hotspot_row=h_row, hotspot=hot, tau_row_to_bins=tau, T=T))

                # Track phase bounds and bin count
                tGl_by_flow[int(fid)] = min(tGl_by_flow[int(fid)], tG)
                tGu_by_flow[int(fid)] = max(tGu_by_flow[int(fid)], tG)
                bins_count_by_flow[int(fid)] += 1

                # Mass weight components and contribution-weighted unit price
                xhat, DH, gH = mass_weight_gH(
                    xG,
                    tG,
                    h_row,
                    hot.bin,
                    self.S_mat,
                    eps=self.params.eps,
                    rolling_occ_by_bin=self.rolling_occ_by_bin,
                    hourly_capacity_matrix=self.hourly_capacity_matrix,
                    bins_per_hour=self.bins_per_hour,
                )

                v_tilde = price_contrib_v_tilde(
                    tG,
                    h_row,
                    hot.bin,
                    tau,
                    self.H_bool,
                    self.S_mat,
                    xG,
                    theta_mask=None,
                    w_sum=self.params.w_sum,
                    w_max=self.params.w_max,
                    kappa=self.params.kappa,
                    eps=self.params.eps,
                    verbose_debug=False,
                    idx_to_tv_id=self.idx_to_tv_id,
                    rolling_occ_by_bin=self.rolling_occ_by_bin,
                    hourly_capacity_matrix=self.hourly_capacity_matrix,
                    bins_per_hour=self.bins_per_hour,
                )

                # Slack base and shifted variants; accumulate sums and track global min rows
                for mins in delta_min_list:
                    shift = delta_to_shift[mins]
                    t_eval = int(tG + shift)
                    if not (0 <= t_eval < T):
                        continue
                    s_val = float(
                        slack_G_at(
                            t_eval,
                            tau,
                            self.S_mat,
                            rolling_occ_by_bin=self.rolling_occ_by_bin,
                            hourly_capacity_matrix=self.hourly_capacity_matrix,
                            bins_per_hour=self.bins_per_hour,
                        )
                    )
                    slack_sums[int(fid)][int(mins)] += float(s_val)

                    # Track global argmin row
                    r_hat, s_min, occ_val, cap_val = self._slack_min_row(t_eval, tau)
                    # If this bin yields a new minimum, update row
                    if s_min is not None and float(s_min) < float(slack_min_vals[int(fid)][int(mins)]):
                        slack_min_vals[int(fid)][int(mins)] = float(s_min)
                        slack_min_rows[int(fid)][int(mins)] = int(r_hat) if r_hat is not None else None
                        slack_min_occ[int(fid)][int(mins)] = float(occ_val) if occ_val is not None else None
                        slack_min_cap[int(fid)][int(mins)] = float(cap_val) if cap_val is not None else None

                # Eligibility (soft) and slack penalty
                rho = slack_penalty(
                    tG,
                    tau,
                    self.S_mat,
                    # S0=self.params.S0,
                    xG=xG,
                    S0_mode=self.params.S0_mode,
                    verbose_debug=False,
                    idx_to_tv_id=self.idx_to_tv_id,
                    rolling_occ_by_bin=self.rolling_occ_by_bin,
                    hourly_capacity_matrix=self.hourly_capacity_matrix,
                    bins_per_hour=self.bins_per_hour,
                )

                # Accumulate
                acc = accs[int(fid)]
                acc["xGH"] += float(xhat)
                acc["DH"] += float(DH)
                acc["gH_sum"] += float(gH)
                acc["v_tilde"] += float(v_tilde)
                # acc["gH_v_tilde_sum"] += float(gH) * float(v_tilde)
                acc["rho"] += float(rho)

        # Build outputs with derived and average variants
        out: Dict[int, FlowFeatures] = {}
        for fid, acc in accs.items():
            bins_count = int(bins_count_by_flow[int(fid)]) or 1
            x_sum = float(acc["xGH"]) 
            d_sum = float(acc["DH"]) 
            g_sum = float(acc["gH_sum"]) 
            v_sum = float(acc["v_tilde"]) 

            flights_meta = getattr(self, "_flights_by_flow", {}).get(int(fid)) or ()
            if flights_meta:
                unique_flights = set()
                for meta in flights_meta:
                    if isinstance(meta, Mapping):
                        fid_val = meta.get("flight_id")
                    else:
                        fid_val = meta
                    if not fid_val:
                        continue
                    unique_flights.add(str(fid_val))
                num_flights = len(unique_flights)
            else:
                num_flights = 0

            # Derived gH = xGH / (xGH + DH) with eps guard
            denom = x_sum + d_sum
            g_derived = float(x_sum / denom) if denom > float(self.params.eps) else 0.0

            # Average of per-bin gH
            g_avg = float(g_sum / float(bins_count)) if bins_count > 0 else 0.0

            # Slack sums and rows
            slack0 = float(slack_sums[int(fid)][0])
            slack15 = float(slack_sums[int(fid)][15])
            slack30 = float(slack_sums[int(fid)][30])
            slack45 = float(slack_sums[int(fid)][45])

            r0 = slack_min_rows[int(fid)][0]
            r15 = slack_min_rows[int(fid)][15]
            r30 = slack_min_rows[int(fid)][30]
            r45 = slack_min_rows[int(fid)][45]

            occ0 = slack_min_occ[int(fid)][0]
            occ15 = slack_min_occ[int(fid)][15]
            occ30 = slack_min_occ[int(fid)][30]
            occ45 = slack_min_occ[int(fid)][45]

            cap0 = slack_min_cap[int(fid)][0]
            cap15 = slack_min_cap[int(fid)][15]
            cap30 = slack_min_cap[int(fid)][30]
            cap45 = slack_min_cap[int(fid)][45]

            out[int(fid)] = FlowFeatures(
                flow_id=int(fid),
                control_tv_id=ctrl_by_flow.get(int(fid)),
                tGl=int(tGl_by_flow[int(fid)]),
                tGu=int(tGu_by_flow[int(fid)]),
                xGH=x_sum,
                DH=d_sum,
                gH_sum=g_sum,
                gH_avg=g_avg,
                gH=g_derived,
                v_tilde=v_sum,
                # gH_v_tilde=float(g_derived * v_sum),
                Slack_G0=slack0,
                Slack_G0_row=(int(r0) if r0 is not None else None),
                Slack_G0_occ=(float(occ0) if occ0 is not None else None),
                Slack_G0_cap=(float(cap0) if cap0 is not None else None),
                Slack_G15=slack15,
                Slack_G15_row=(int(r15) if r15 is not None else None),
                Slack_G15_occ=(float(occ15) if occ15 is not None else None),
                Slack_G15_cap=(float(cap15) if cap15 is not None else None),
                Slack_G30=slack30,
                Slack_G30_row=(int(r30) if r30 is not None else None),
                Slack_G30_occ=(float(occ30) if occ30 is not None else None),
                Slack_G30_cap=(float(cap30) if cap30 is not None else None),
                Slack_G45=slack45,
                Slack_G45_row=(int(r45) if r45 is not None else None),
                Slack_G45_occ=(float(occ45) if occ45 is not None else None),
                Slack_G45_cap=(float(cap45) if cap45 is not None else None),
                rho=float(acc["rho"]),
                bins_count=bins_count,
                num_flights=int(num_flights),
            )

        return out
