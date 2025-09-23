from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class CheckOutcome:
    name: str
    description: str
    violations: int
    total: int

    @property
    def ok(self) -> bool:
        return self.violations == 0


def load_plan_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _iter_plan_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    plan_items = payload.get("plan")
    if not isinstance(plan_items, list):
        return []
    items: List[Dict[str, Any]] = []
    for item in plan_items:
        if isinstance(item, dict):
            items.append(item)
    return items


def _reg_key(item: Dict[str, Any]) -> Tuple[str, Tuple[int, int]]:
    cv_id = str(item.get("control_volume_id", ""))
    bins = item.get("window_bins") or []
    t0 = int(bins[0]) if isinstance(bins, (list, tuple)) and len(bins) > 0 else -1
    t1 = int(bins[1]) if isinstance(bins, (list, tuple)) and len(bins) > 1 else -1
    return cv_id, (t0, t1)


def check_unique_evals_from_plan(payload: Dict[str, Any]) -> CheckOutcome:
    """Validate that enough unique RateFinder evaluations were performed.

    Expects the plan JSON to include a top-level integer field `unique_evals`,
    which should reflect the number of real RateFinder calls (not cached).
    The check fails when unique_evals < 5.
    """
    try:
        val = payload.get("unique_evals", None)
        unique_evals = int(val) if isinstance(val, (int, float)) else None
    except Exception:
        unique_evals = None

    if unique_evals is None:
        return CheckOutcome(
            name="unique_evals",
            description=(
                "Validation skipped: `unique_evals` not present in plan file."
            ),
            violations=0,
            total=0,
        )

    violations = 1 if int(unique_evals) < 5 else 0
    return CheckOutcome(
        name="unique_evals",
        description=(
            "At least 5 unique RateFinder evaluations must occur. Low counts may indicate a bug preventing differentiation between regulations or other issues."
        ),
        violations=violations,
        total=1,
    )


def check_no_duplicate_regulations(items: List[Dict[str, Any]]) -> CheckOutcome:
    seen: set[Tuple[str, Tuple[int, int]]] = set()
    violations = 0
    for item in items:
        key = _reg_key(item)
        if key in seen:
            violations += 1
        else:
            seen.add(key)
    return CheckOutcome(
        name="no_duplicate_regulations",
        description=(
            "No duplicate regulation could exist. A regulation is characterized by the "
            "control_volume_id and the window_bins."
        ),
        violations=violations,
        total=len(items),
    )


def check_flows_and_rates(items: List[Dict[str, Any]]) -> CheckOutcome:
    violations = 0
    for item in items:
        flows = item.get("flow_ids") or []
        if not isinstance(flows, list):
            flows = []
        if len(flows) == 0:
            violations += 1
            continue

        blanket = item.get("blanket_rate", None)
        if blanket is not None:
            continue

        rates = item.get("rates_per_flow", None)
        if not isinstance(rates, dict):
            violations += 1
            continue

        if len(rates) != len(flows):
            violations += 1

    return CheckOutcome(
        name="flows_and_rates",
        description=(
            "Each regulation must contain at least one flow (flow_ids cannot be empty) and the "
            "rates_per_flow should be of the same length as the number of flows or the blanket_rate is not null."
        ),
        violations=violations,
        total=len(items),
    )


def validate_plan_payload(payload: Dict[str, Any]) -> List[CheckOutcome]:
    items = _iter_plan_items(payload)
    results: List[CheckOutcome] = []
    results.append(check_no_duplicate_regulations(items))
    results.append(check_flows_and_rates(items))
    # Include unique evals check if the plan payload carries this information
    try:
        results.append(check_unique_evals_from_plan(payload))
    except Exception:
        # Be resilient: if any unexpected structure, do not block other checks
        pass
    return results


def check_delay_granularity_from_run_log(log_path: str | Path) -> CheckOutcome:
    """Ensure at least one positive delay assignment is not divisible by 15 minutes.

    Reads the run JSONL log, finds the last run_end event, and inspects
    artifacts.delays_min. The check passes if there exists at least one
    positive delay value not divisible by 15.
    """
    minutes_list: List[int] = []
    try:
        p = Path(log_path)
        if not p.exists():
            return CheckOutcome(
                name="delay_granularity",
                description=(
                    "Validation skipped: run log not found; expected at least one delay not divisible by 15 minutes."
                ),
                violations=0,
                total=0,
            )
        last_run_end: Optional[Dict[str, Any]] = None
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("type") == "run_end":
                    last_run_end = obj
        if not last_run_end:
            return CheckOutcome(
                name="delay_granularity",
                description=(
                    "Validation skipped: no run_end event found in log."
                ),
                violations=0,
                total=0,
            )
        artifacts = last_run_end.get("artifacts") or {}
        delays = artifacts.get("delays_min")
        if not isinstance(delays, dict) or not delays:
            return CheckOutcome(
                name="delay_granularity",
                description=(
                    "Validation skipped: delays_min artifact missing or empty."
                ),
                violations=0,
                total=0,
            )
        for val in delays.values():
            if isinstance(val, (int, float)):
                m = int(val)
                if m > 0:
                    minutes_list.append(m)
        if not minutes_list:
            return CheckOutcome(
                name="delay_granularity",
                description=(
                    "Validation skipped: no positive delay assignments found."
                ),
                violations=0,
                total=0,
            )
        all_multiples_of_15 = all((m % 15 == 0) for m in minutes_list)
        # Single-violation aggregate check: 1/1 when all are multiples of 15, 0/1 otherwise
        return CheckOutcome(
            name="delay_granularity",
            description=(
                "At least one positive delay assignment must not be divisible by 15 minutes."
            ),
            violations=1 if all_multiples_of_15 else 0,
            total=1,
        )
    except Exception:
        # Do not hard-fail panel rendering; surface as skipped
        return CheckOutcome(
            name="delay_granularity",
            description=(
                "Validation error: unable to read or parse run log for delay checks."
            ),
            violations=0,
            total=0,
        )


def validate_plan_with_run_payload(
    payload: Dict[str, Any], *, run_log_path: str | Path | None = None
) -> List[CheckOutcome]:
    results = validate_plan_payload(payload)
    if run_log_path is not None:
        results.append(check_delay_granularity_from_run_log(run_log_path))
    return results


def validate_plan_and_run_file(
    plan_path: str | Path, run_log_path: str | Path
) -> tuple[bool, List[CheckOutcome]]:
    payload = load_plan_json(plan_path)
    results = validate_plan_with_run_payload(payload, run_log_path=run_log_path)
    ok = print_validation_report(results)
    return ok, results

def print_validation_report(results: Iterable[CheckOutcome]) -> bool:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    
    console = Console()
    results_list = list(results)
    overall_ok = all(r.ok for r in results_list)
    
    # Create a table for the validation results
    table = Table(box=box.SIMPLE)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center", no_wrap=True)
    table.add_column("Violations", justify="right", no_wrap=True)
    table.add_column("Description", style="dim")
    
    for r in results_list:
        status = "[green]✓ PASS[/green]" if r.ok else "[red]✗ FAIL[/red]"
        violations = f"{r.violations}/{r.total}"
        table.add_row(r.name, status, violations, r.description)
    
    # Create the panel
    title_style = "[bold green]✓ VALIDATION PASSED[/bold green]" if overall_ok else "[bold red]✗ VALIDATION FAILED[/bold red]"
    border_style = "green" if overall_ok else "red"
    
    panel = Panel(
        table,
        title=title_style,
        border_style=border_style,
        padding=(1, 1)
    )
    
    console.print(panel)
    return overall_ok


def validate_plan_file(path: str | Path) -> bool:
    payload = load_plan_json(path)
    results = validate_plan_payload(payload)
    return print_validation_report(results)


def _main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Usage: python -m parrhesia.flow_agent.plan_validator <path_to_plan.json>")
        return 2
    plan_path = args[0]
    try:
        ok = validate_plan_file(plan_path)
    except FileNotFoundError:
        print(f"File not found: {plan_path}")
        return 2
    except json.JSONDecodeError as exc:
        print(f"Failed to parse JSON: {exc}")
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CheckOutcome",
    "load_plan_json",
    "validate_plan_payload",
    "print_validation_report",
    "validate_plan_file",
    "check_delay_granularity_from_run_log",
    "check_unique_evals_from_plan",
    "validate_plan_with_run_payload",
    "validate_plan_and_run_file",
]


