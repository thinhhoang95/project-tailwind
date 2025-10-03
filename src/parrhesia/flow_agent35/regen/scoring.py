"""Flow pruning and scoring utilities."""
from __future__ import annotations

from typing import List, Mapping, Sequence

from parrhesia.metaopt.feats.flow_features import FlowFeatures

from .features_bridge import coverage_i
from .types import FlowDiagnostics, FlowScore, FlowScoreWeights, RegenConfig


def _baseline_rate_from_features(feature: FlowFeatures, *, bins_per_hour: int) -> float:
    bins_count = max(1, int(feature.bins_count))
    avg_bin_demand = float(feature.xGH) / float(bins_count)
    return avg_bin_demand * float(bins_per_hour)


def prune_flows(
    *,
    features: Mapping[int, FlowFeatures],
    config: RegenConfig,
    bins_per_hour: int,
    verbose_debug: bool = False,
) -> List[int]:
    """Return flow ids that pass the eligibility filters."""

    eligible: List[int] = []
    slack_min = float(config.slack_min)
    for flow_id, feat in features.items():
        if feat.num_flights < config.min_num_flights:
            if verbose_debug:
                print(f'Pruning flow {flow_id} because num_flights is {feat.num_flights}')
            continue
        if float(feat.xGH) <= 0.0:
            if verbose_debug:
                print(f'Pruning flow {flow_id} because xGH is 0')
            continue
        r0_i = _baseline_rate_from_features(feat, bins_per_hour=bins_per_hour)
        if r0_i <= 0.0:
            if verbose_debug:
                print(f'Pruning flow {flow_id} because r0_i is 0')
            continue
        if float(feat.gH) <= float(config.g_min):
            if verbose_debug:
                print(f'Pruning flow {flow_id} because gH is {float(feat.gH)}')
            continue
        if float(feat.rho) >= float(config.rho_max):
            if verbose_debug:
                print(f'Pruning flow {flow_id} because rho is {float(feat.rho)}')
            continue
        slack15 = float(getattr(feat, "Slack_G15", 0.0) or 0.0)
        if slack15 <= slack_min:
            if verbose_debug:
                print(f'Pruning flow {flow_id} because slack15 is {slack15}')
            slack30 = float(getattr(feat, "Slack_G30", 0.0) or 0.0)
            slack45 = float(getattr(feat, "Slack_G45", 0.0) or 0.0)
            if max(slack30, slack45) <= slack_min:
                if verbose_debug:
                    print(f'Pruning flow {flow_id} because slack30 or slack45 is {max(slack30, slack45)}')
                continue
        eligible.append(int(flow_id))
    return eligible


def score_flows(
    *,
    eligible_flows: Sequence[int],
    features: Mapping[int, FlowFeatures],
    weights: FlowScoreWeights,
    indexer,
    timebins_h: Sequence[int],
    verbose_debug: bool = False,
) -> List[FlowScore]:
    """Score eligible flows and attach diagnostics."""

    bins_per_hour = int(indexer.rolling_window_size())
    scores: List[FlowScore] = []
    for flow_id in eligible_flows:
        feat = features.get(int(flow_id))
        if feat is None:
            continue
        r0_i = _baseline_rate_from_features(feat, bins_per_hour=bins_per_hour)
        cov = coverage_i(tGl=int(feat.tGl), tGu=int(feat.tGu), timebins_h=timebins_h, indexer=indexer)
        slack15 = float(getattr(feat, "Slack_G15", 0.0) or 0.0)
        slack30 = float(getattr(feat, "Slack_G30", 0.0) or 0.0)
        slack45 = float(getattr(feat, "Slack_G45", 0.0) or 0.0)
        score_val = (
            weights.w1 * float(feat.gH)
            # + weights.w2 * float(feat.gH_v_tilde)
            + weights.w2 * float(feat.v_tilde)
            + weights.w3 * slack15
            + weights.w4 * slack30
            - weights.w5 * float(feat.rho)
            + weights.w6 * cov
        )

        # Choose one of the following: clamp to 0 or use the raw score
        # If slack is very negative (we allow negative slack for robustness), then clamping to 0 will be too restrictive
        # score_val = max(0.0, float(score_val)) 
        score_val = float(score_val)

        diagnostics = FlowDiagnostics(
            gH=float(feat.gH),
            # gH_v_tilde=float(feat.gH_v_tilde),
            v_tilde=float(feat.v_tilde),
            rho=float(feat.rho),
            slack15=slack15,
            slack30=slack30,
            slack45=slack45,
            coverage=float(cov),
            r0_i=float(r0_i),
            xGH=float(feat.xGH),
            DH=float(feat.DH),
            tGl=int(feat.tGl),
            tGu=int(feat.tGu),
            bins_count=int(feat.bins_count),
            num_flights=int(getattr(feat, "num_flights", 0)),
        )
        scores.append(
            FlowScore(
                flow_id=int(flow_id),
                control_tv_id=str(feat.control_tv_id) if feat.control_tv_id is not None else None,
                score=score_val,
                diagnostics=diagnostics,
                num_flights=int(getattr(feat, "num_flights", 0)),
            )
        )
    scores.sort(key=lambda fs: fs.score, reverse=True)

    if verbose_debug:
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:  # pragma: no cover - optional dependency
            print("Rich is not installed; skipping verbose debug table.")
        else:
            console = Console()
            table = Table(title="Flow Scores", highlight=True, expand=True)
            table.add_column("Flow ID", style="cyan", justify="right")
            table.add_column("Control TV", style="cyan")
            table.add_column("Score", style="green", justify="right")
            table.add_column("gH", justify="right")
            table.add_column("v_tilde", justify="right")
            table.add_column("rho", justify="right")
            table.add_column("Slack15", justify="right")
            table.add_column("Slack30", justify="right")
            table.add_column("Slack45", justify="right")
            table.add_column("Coverage", justify="right")
            table.add_column("r0_i", justify="right")
            table.add_column("xGH", justify="right")
            table.add_column("DH", justify="right")
            table.add_column("tGl", justify="right")
            table.add_column("tGu", justify="right")
            table.add_column("Bins", justify="right")
            table.add_column("Flights", justify="right")

            for score in scores:
                diag = score.diagnostics
                table.add_row(
                    str(score.flow_id),
                    str(score.control_tv_id) if score.control_tv_id is not None else "-",
                    f"{score.score:.3f}",
                    f"{diag.gH:.3f}",
                    f"{diag.v_tilde:.3f}",
                    f"{diag.rho:.3f}",
                    f"{diag.slack15:.3f}",
                    f"{diag.slack30:.3f}",
                    f"{diag.slack45:.3f}",
                    f"{diag.coverage:.3f}",
                    f"{diag.r0_i:.3f}",
                    f"{diag.xGH:.3f}",
                    f"{diag.DH:.3f}",
                    str(diag.tGl),
                    str(diag.tGu),
                    str(diag.bins_count),
                    str(score.num_flights),
                )
            console.print(table)

    return scores
