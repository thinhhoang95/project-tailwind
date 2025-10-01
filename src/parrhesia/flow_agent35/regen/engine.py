"""Main entry-point for generating regulation proposals."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import logging
import numpy as np

from .bundles import build_candidate_bundles
from .config import resolve_config, resolve_weights
from .exceedance import compute_hotspot_exceedance
from .features_bridge import (
    baseline_demand_by_flow_from_payload,
    build_flow_features_extractor,
    extract_features_for_flows,
)
from .predict import (
    apply_regulation_to_schedule,
    baseline_schedule_from_context,
    build_local_context,
    compute_delta_deficit_per_hour,
    score_pair,
)
from .rates import compute_e_target, occupancy_to_entrance, rate_cuts_for_bundle
from .scoring import prune_flows, score_flows
from .search import local_search_variants
from .types import (
    Bundle,
    BundleVariant,
    PredictedImprovement,
    Proposal,
    RateCut,
    RegenConfig,
    FlowScoreWeights,
    Window,
)
from .window import select_window_for_bundle
from parrhesia.optim.ripple import compute_auto_ripple_cells


AUTO_RIPPLE_DILATION_BINS = 2


logger = logging.getLogger(__name__)


def _flow_ids_from_rates(rates: Sequence[RateCut]) -> List[int]:
    return sorted({int(rc.flow_id) for rc in rates})


def _window_overlap_ratio(a: Window, b: Window) -> float:
    start = max(a.start_bin, b.start_bin)
    end = min(a.end_bin, b.end_bin)
    if end < start:
        return 0.0
    union = max(a.end_bin, b.end_bin) - min(a.start_bin, b.start_bin) + 1
    if union <= 0:
        return 0.0
    return float(end - start + 1) / float(union)


def _bundle_overlap_score(candidate: BundleVariant, selected: Sequence[BundleVariant]) -> float:
    if not selected:
        return 0.0
    candidate_set = set(_flow_ids_from_rates(candidate.rates))
    max_overlap = 0.0
    for other in selected:
        other_set = set(_flow_ids_from_rates(other.rates))
        union = len(candidate_set | other_set)
        inter = len(candidate_set & other_set)
        jaccard = float(inter) / float(union) if union else 0.0
        time_overlap = _window_overlap_ratio(candidate.window, other.window)
        overlap = 0.5 * jaccard + 0.5 * time_overlap
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap


def _build_target_cells(hotspot_tv: str, timebins_h: Sequence[int]) -> List[Tuple[str, int]]:
    return [(str(hotspot_tv), int(b)) for b in timebins_h]


def _log_capacity_snapshot(
    label: str,
    tv_ids: Sequence[str],
    capacities_by_tv: Mapping[str, np.ndarray],
    timebins: Sequence[int],
    max_entries: int = 8,
) -> None:
    ids_unique = list(dict.fromkeys(str(tv) for tv in tv_ids))
    if not ids_unique:
        print("Regen: capacity snapshot [%s] skipped (no TVs)", label)
        return
    truncated = len(ids_unique) > max_entries
    ids_for_log = ids_unique[:max_entries]
    entries: List[str] = []
    for tv in ids_for_log:
        arr = capacities_by_tv.get(str(tv))
        if arr is None:
            entries.append(f"{tv}:missing")
            continue
        arr_np = np.asarray(arr, dtype=np.float64)
        if arr_np.size == 0:
            entries.append(f"{tv}:empty")
            continue
        if timebins:
            valid_bins = [b for b in timebins if 0 <= int(b) < arr_np.size]
            slice_vals = arr_np[valid_bins] if valid_bins else np.asarray([], dtype=np.float64)
        else:
            slice_vals = arr_np
        if slice_vals.size == 0:
            entries.append(f"{tv}:no-window-data")
            continue
        entries.append(
            f"{tv}:min={float(slice_vals.min()):.1f},max={float(slice_vals.max()):.1f},mean={float(slice_vals.mean()):.1f}"
        )
    if truncated:
        entries.append(f"... {len(ids_unique) - max_entries} more")
    print("Regen: capacity snapshot [%s] %s", label, "; ".join(entries))


def _tvs_traversed_by_flows(*, flights_by_flow: Mapping[int, Sequence[Mapping[str, Any]]], flight_list: Any, indexer) -> List[str]:
    """Return all TV ids traversed by at least one flight among the provided flows.

    Decodes each flight's occupancy intervals via ``tvtw_index`` -> (tv_row, bin)
    using the indexer's number of time bins per TV and the flight list's
    ``idx_to_tv_id`` mapping.
    """
    tvs: set[str] = set()
    if not flights_by_flow:
        return []
    # Prefer mapping from the flight_list; fall back to indexer if available
    idx_to_tv_id = getattr(flight_list, "idx_to_tv_id", None) or getattr(indexer, "idx_to_tv_id", None) or {}
    try:
        bins_per_tv = int(getattr(indexer, "num_time_bins"))
    except Exception:
        # Conservative fallback; if unknown, we cannot decode tvtw -> tv
        bins_per_tv = None  # type: ignore
    fm = getattr(flight_list, "flight_metadata", {}) or {}
    for _flow_id, specs in flights_by_flow.items():
        for sp in (specs or ()):  # each spec is a mapping with 'flight_id'
            fid = sp.get("flight_id") if isinstance(sp, dict) else None
            if fid is None:
                continue
            meta = fm.get(str(fid)) or {}
            intervals = meta.get("occupancy_intervals") or []
            for iv in intervals:
                tvtw_raw = iv.get("tvtw_index") if isinstance(iv, dict) else None
                if tvtw_raw is None:
                    continue
                try:
                    tvtw_idx = int(tvtw_raw)
                except Exception:
                    continue
                if bins_per_tv is None or bins_per_tv <= 0:
                    # Without bins_per_tv we cannot decode; skip safely
                    continue
                tv_row = int(tvtw_idx) // int(bins_per_tv)
                tv_id = idx_to_tv_id.get(int(tv_row)) if isinstance(idx_to_tv_id, dict) else None
                if tv_id is not None:
                    tvs.add(str(tv_id))
    return sorted(tvs)


def _build_tv_filter(
    hotspot_tv: str,
    bundles: Sequence[Bundle],
    *,
    flights_by_flow: Mapping[int, Sequence[Mapping[str, Any]]],
    flight_list: Any,
    indexer,
) -> List[str]:
    """TVs used for objective evaluation.

    Includes:
      - the hotspot itself,
      - all control TVs of flows in the candidate bundles,
      - and all TVs traversed by at least one flight of those flows (network impact).
    """
    tvs = {str(hotspot_tv)}
    for bundle in bundles:
        for fs in bundle.flows:
            if fs.control_tv_id:
                tvs.add(str(fs.control_tv_id))
    # Add all TVs traversed by flights of the flows under consideration
    traversed = _tvs_traversed_by_flows(
        flights_by_flow=flights_by_flow,
        flight_list=flight_list,
        indexer=indexer,
    )
    tvs.update(traversed)
    return sorted(tvs)


def propose_regulations_for_hotspot(
    *,
    indexer,
    flight_list,
    capacities_by_tv: Mapping[str, np.ndarray],
    travel_minutes_map: Mapping[str, Mapping[str, float]],
    hotspot_tv: str,
    timebins_h: Sequence[int],
    flows_payload: Optional[Mapping[str, Any]] = None,
    flow_to_flights: Optional[Mapping[str, Sequence[str]]] = None,
    weights: Optional[FlowScoreWeights] = None,
    config: Optional[RegenConfig] = None,
) -> List[Proposal]:
    cfg = resolve_config(config)
    wts = resolve_weights(weights)

    extractor = build_flow_features_extractor(
        indexer=indexer,
        flight_list=flight_list,
        capacities_by_tv=capacities_by_tv,
        travel_minutes_map=travel_minutes_map,
        autotrim_from_ctrl_to_hotspot=cfg.autotrim_from_ctrl_to_hotspot,
    )
    timebins_seq = [int(b) for b in timebins_h]
    features = extract_features_for_flows(
        extractor,
        hotspot_tv=hotspot_tv,
        timebins=timebins_seq,
        flows_payload=flows_payload,
    )
    exceedance_stats = compute_hotspot_exceedance(
        indexer=indexer,
        flight_list=flight_list,
        capacities_by_tv=capacities_by_tv,
        hotspot_tv=hotspot_tv,
        timebins_h=timebins_seq,
        caches=extractor.caches,
    )
    # D_vec only contains the exceedance for the hotspot TV and the designated timebins
    D_vec = exceedance_stats.get("D_vec", np.zeros(1, dtype=float))
    # Cut target E_target
    # TV scope: hotspot TV only.
    # Time scope: exactly the provided timebins_h for the hotspot.
    # Flow scope: aggregated rolling occupancy at the hotspot (not flow-specific); exceedance is per-bin (rolling occupancy − hourly capacity); optionally converted from occupancy to entrance.
    E_target_occ = compute_e_target(D_vec, mode=cfg.e_target_mode, fallback_to_peak=cfg.fallback_to_peak)
    if cfg.convert_occupancy_to_entrance:
        E_target = occupancy_to_entrance(
            E_target_occ,
            dwell_minutes=cfg.dwell_minutes,
            alpha=cfg.alpha_occupancy_to_entrance,
        )
    else:
        E_target = float(E_target_occ)
    if E_target <= 0.0:
        if cfg.raise_on_edge_cases:
            raise ValueError("Hotspot exceedance target is zero; nothing to regulate")
        return []

    bins_per_hour = int(indexer.rolling_window_size())
    eligible_ids = prune_flows(features=features, config=cfg, bins_per_hour=bins_per_hour)
    if not eligible_ids:
        if cfg.raise_on_edge_cases:
            raise ValueError("No eligible flows after pruning")
        return []
    scored_flows = score_flows(
        eligible_flows=eligible_ids,
        features=features,
        weights=wts,
        indexer=indexer,
        timebins_h=timebins_h,
    )
    if not scored_flows:
        if cfg.raise_on_edge_cases:
            raise ValueError("No flows scored for regulation")
        return []

    demand_by_flow, flights_by_flow = baseline_demand_by_flow_from_payload(
        indexer,
        flows_payload=flows_payload,
        flow_to_flights=flow_to_flights,
        flight_list=flight_list,
        features=features,
    )
    flights_by_flow = {int(fid): tuple(specs) for fid, specs in flights_by_flow.items()}

    bundles = build_candidate_bundles(
        scored_flows,
        max_bundle_size=cfg.max_bundle_size,
        distinct_controls_required=cfg.distinct_controls_required,
    )
    if not bundles:
        if cfg.raise_on_edge_cases:
            raise ValueError("No candidate bundles constructed")
        return []

    # Optional: add two shoulder timebins to cover the "dumping" effect from unreleased flights in the queue, not used for now
    # timebins_seq_with_shoulder = timebins_seq + [timebins_seq[-1] + 1, timebins_seq[-1] + 2] # add two shoulder timebins to cover the "dumping" effect from unreleased flights in the queue
    
    # Target cells: hotspot TV across designated hotspot timebins
    target_cells = _build_target_cells(hotspot_tv, timebins_seq)
    _log_capacity_snapshot("hotspot", [hotspot_tv], capacities_by_tv, timebins_seq)

    candidates: List[Tuple[BundleVariant, PredictedImprovement, Dict[str, Any]]] = []
    total_bins = int(getattr(indexer, "num_time_bins"))


    # Each bundle is one regulation proposal that could contain one, two or more flows
    for bundle in bundles:
        # 1) Select evaluation window for this bundle
        window = select_window_for_bundle(
            bundle,
            bins_per_hour=bins_per_hour,
            total_bins=total_bins,
            margin_hours=cfg.window_margin_hours,
            min_window_hours=cfg.min_window_hours,
        )
        # 2) Build base rate cuts for this bundle
        base_cuts = rate_cuts_for_bundle(bundle, E_target=E_target, bins_per_hour=bins_per_hour)
        if not base_cuts:
            continue
        # 3) Compute per-bundle ripple cells from flights of flows in the bundle only,
        #    then derive tv_filter = hotspot ∪ ripple TVs.
        bundle_flow_ids = [int(fs.flow_id) for fs in bundle.flows]
        bundle_flight_ids: List[str] = []
        for fid in bundle_flow_ids:
            for sp in (flights_by_flow.get(int(fid)) or ()):  # each spec is a mapping with 'flight_id'
                _sp_fid = sp.get("flight_id") if isinstance(sp, dict) else None
                if _sp_fid is None:
                    continue
                bundle_flight_ids.append(str(_sp_fid))
        bundle_ripple_cells = compute_auto_ripple_cells(
            indexer=indexer,
            flight_list=flight_list,
            flight_ids=sorted(set(bundle_flight_ids)),
            window_bins=AUTO_RIPPLE_DILATION_BINS,
        )
        bundle_ripple_tvs = sorted({str(tv) for (tv, _b) in bundle_ripple_cells})
        bundle_tv_filter = sorted({str(hotspot_tv)} | set(bundle_ripple_tvs))
        _log_capacity_snapshot("tv_filter", bundle_tv_filter, capacities_by_tv, timebins_seq)

        # 4) Build a localized scoring context per-bundle so that ripple/target TVs are correct.
        #    Context mirrors the simulated annealing scoring
        context_bundle = build_local_context(
            indexer=indexer,
            flight_list=flight_list,
            capacities_by_tv=capacities_by_tv,
            target_cells=target_cells,
            flights_by_flow=flights_by_flow,
            weights=None,
            tv_filter=bundle_tv_filter,
            ripple_cells=bundle_ripple_cells,
        )
        baseline_schedule = baseline_schedule_from_context(context_bundle)
        context_flow_ids = set(int(fid) for fid in baseline_schedule.keys())
        demand_for_context = {
            int(fid): np.asarray(arr, dtype=np.int64)
            for fid, arr in demand_by_flow.items()
            if int(fid) in context_flow_ids
        }
        variants = local_search_variants(
            bundle,
            base_cuts,
            steps=cfg.local_search_steps,
            max_variants=cfg.max_variants_per_bundle,
            use_percent=cfg.local_search_use_percent,
            percent_lower=cfg.local_search_percent_lower,
            percent_upper=cfg.local_search_percent_upper,
            percent_step=cfg.local_search_percent_step,
        )
        best_variant: Optional[Tuple[BundleVariant, PredictedImprovement, Dict[str, Any]]] = None
        for rates in variants:
            regulated_schedule = apply_regulation_to_schedule(
                baseline_schedule,
                demand_for_context,
                rates=rates,
                indexer=indexer,
                window=window,
            )
            (
                score_before,
                score_after,
                occ_before,
                occ_after,
                components_before,
                components_after,
            ) = score_pair(
                baseline_schedule,
                regulated_schedule,
                flights_by_flow=flights_by_flow,
                capacities_by_tv=capacities_by_tv,
                flight_list=flight_list,
                context=context_bundle,
            )
            delta_objective = score_before - score_after
            delta_deficit = compute_delta_deficit_per_hour(
                occ_before,
                occ_after,
                capacities_by_tv=capacities_by_tv,
                target_cells=target_cells,
                indexer=indexer,
                window=window,
            )
            improvement = PredictedImprovement(
                delta_deficit_per_hour=float(delta_deficit),
                delta_objective_score=float(delta_objective),
            )
            bundle_variant = BundleVariant(bundle=bundle, window=window, rates=list(rates))
            diagnostics = {
                "score_before": float(score_before),
                "score_after": float(score_after),
                "score_components_before": components_before,
                "score_components_after": components_after,
                "weights_used": dict(bundle.weights_by_flow),
                "E_target": float(E_target),
                "E_target_occupancy": float(E_target_occ),
                # Include per-bundle ripple context used for evaluation
                "ripple_tvs": list(bundle_ripple_tvs),
                "ripple_cells": [(str(tv), int(b)) for (tv, b) in bundle_ripple_cells],
            }
            if best_variant is None or improvement.delta_objective_score > best_variant[1].delta_objective_score:
                best_variant = (bundle_variant, improvement, diagnostics)
        if best_variant is not None:
            candidates.append(best_variant)

    if not candidates:
        if cfg.raise_on_edge_cases:
            raise ValueError("No viable regulation variants produced")
        return []

    # Diversity-aware selection
    selected: List[Tuple[BundleVariant, PredictedImprovement, Dict[str, Any], float]] = []
    remaining = sorted(
        candidates,
        key=lambda item: item[1].delta_objective_score,
        reverse=True,
    )
    for candidate in remaining:
        if len(selected) >= cfg.k_proposals:
            break
        bundle_variant, improvement, diag = candidate
        overlap = _bundle_overlap_score(bundle_variant, [bv for bv, _, _, _ in selected])
        penalty = cfg.diversity_alpha * overlap
        final_score = improvement.delta_objective_score - penalty
        selected.append((bundle_variant, improvement, {**diag, "diversity_penalty": penalty}, final_score))
    selected.sort(key=lambda item: item[3], reverse=True)

    proposals: List[Proposal] = []
    for bundle_variant, improvement, diag, final_score in selected:
        flows_info: List[Dict[str, Any]] = []
        weights_dict = dict(bundle_variant.bundle.weights_by_flow)
        per_flow_diag: Dict[int, Dict[str, Any]] = {}
        for fs in bundle_variant.bundle.flows:
            diag_map = {
                "gH": fs.diagnostics.gH,
                # "gH_v_tilde": fs.diagnostics.gH_v_tilde,
                "v_tilde": fs.diagnostics.v_tilde,
                "rho": fs.diagnostics.rho,
                "slack15": fs.diagnostics.slack15,
                "slack30": fs.diagnostics.slack30,
                "slack45": fs.diagnostics.slack45,
                "coverage": fs.diagnostics.coverage,
                "r0_i": fs.diagnostics.r0_i,
                "xGH": fs.diagnostics.xGH,
                "DH": fs.diagnostics.DH,
                "tGl": fs.diagnostics.tGl,
                "tGu": fs.diagnostics.tGu,
                "bins_count": fs.diagnostics.bins_count,
                "num_flights": fs.num_flights,
                "weight": float(weights_dict.get(fs.flow_id, 0.0)),
            }
            per_flow_diag[int(fs.flow_id)] = diag_map
        rates_map = {rc.flow_id: rc for rc in bundle_variant.rates}
        for flow_id, info in per_flow_diag.items():
            rc = rates_map.get(flow_id)
            if rc is None:
                continue
            flows_info.append(
                {
                    "flow_id": int(flow_id),
                    "control_tv_id": next(
                        (fs.control_tv_id for fs in bundle_variant.bundle.flows if fs.flow_id == flow_id),
                        None,
                    ),
                    "R_i": int(rc.allowed_rate_R),
                    "r0_i": float(rc.baseline_rate_r0),
                    "lambda_cut_i": int(rc.cut_per_hour_lambda),
                    "num_flights": info.get("num_flights", 0),
                }
            )
        flows_info.sort(key=lambda item: item["flow_id"])
        proposal_diag = dict(diag)
        proposal_diag.update(
            {
                "per_flow": per_flow_diag,
                "weights_used": weights_dict,
                "ranking_score": float(final_score),
            }
        )
        ctrl_volume = next((fs.control_tv_id for fs in bundle_variant.bundle.flows if fs.control_tv_id), hotspot_tv)
        proposals.append(
            Proposal(
                hotspot_id=str(hotspot_tv),
                controlled_volume=str(ctrl_volume),
                window=bundle_variant.window,
                flows_info=flows_info,
                predicted_improvement=improvement,
                diagnostics=proposal_diag,
                target_cells=[(str(tv), int(b)) for (tv, b) in target_cells],
                ripple_cells=list(proposal_diag.get("ripple_cells", [])),
                target_tvs=[str(hotspot_tv)],
                ripple_tvs=list(proposal_diag.get("ripple_tvs", [])),
            )
        )
    return proposals
