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
    return results

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
]


