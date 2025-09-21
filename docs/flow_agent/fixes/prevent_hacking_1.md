I’ll prevent “empty regulations” at three layers: during commit, when the agent appends to the plan, and on export. I’ll sanitize rates and skip commits with no effective rates or no flows.

Edits:

```107:126:src/parrhesia/flow_agent/transition.py
        if isinstance(action, CommitRegulation):
            guard_can_commit(next_state)
            context = next_state.hotspot_context
            assert context is not None
            # Sanitize committed rates and skip no-op commits (no positive rates)
            rates = action.committed_rates
            rates_to_store = None
            valid = False
            if isinstance(rates, dict):
                cleaned = {}
                for k, v in (rates or {}).items():
                    try:
                        iv = int(v)
                    except Exception:
                        iv = 0
                    if iv > 0:
                        cleaned[str(k)] = iv
                if cleaned:
                    rates_to_store = cleaned
                    valid = True
            elif rates is not None:
                try:
                    iv = int(round(float(rates)))
                except Exception:
                    iv = 0
                if iv > 0:
                    rates_to_store = iv
                    valid = True

            if valid:
                regulation = RegulationSpec(
                    control_volume_id=context.control_volume_id,
                    window_bins=context.window_bins,
                    flow_ids=context.selected_flow_ids,
                    mode=context.mode,
                    committed_rates=rates_to_store,
                    diagnostics=dict(action.diagnostics),
                )
                next_state.plan.append(regulation)

            next_state.reset_hotspot(next_stage="idle")
            next_state.metadata.pop("awaiting_commit", None)
            is_commit = True
            return next_state, is_commit, False
```

```168:206:src/parrhesia/flow_agent/agent.py
            # Materialize regulation and append to plan without relying on stage guards
            # Extract details from diagnostics
            diag = (commit_action.diagnostics or {}).get("rate_finder", {})
            ctrl = str(diag.get("control_volume_id")) if diag.get("control_volume_id") is not None else None
            win_list = diag.get("window_bins") or []
            try:
                wb = (int(win_list[0]), int(win_list[1]))
            except Exception:
                wb = (0, 1)
            mode = str(diag.get("mode", "per_flow"))
            # Flow ids for the regulation
            flow_ids: Tuple[str, ...]
            if isinstance(commit_action.committed_rates, dict):
                flow_ids = tuple(sorted(str(k) for k in commit_action.committed_rates.keys()))
            else:
                entrants = diag.get("entrants_by_flow", {}) or {}
                flow_ids = tuple(sorted(str(k) for k in entrants.keys()))

            from .state import RegulationSpec  # local import to avoid cycles at module import time

            if ctrl is None:
                # As a fallback, skip appending if control volume is unknown
                break

            # Validate and sanitize committed rates; skip empty/no-op commits
            rates = commit_action.committed_rates
            valid = False
            rates_to_store: Optional[Dict[str, int] | int] = None
            if isinstance(rates, dict):
                cleaned: Dict[str, int] = {}
                for k, v in (rates or {}).items():
                    try:
                        iv = int(v)
                    except Exception:
                        iv = 0
                    if iv > 0:
                        cleaned[str(k)] = iv
                if cleaned and len(flow_ids) > 0:
                    valid = True
                    rates_to_store = cleaned
            else:
                try:
                    iv = int(round(float(rates))) if rates is not None else 0
                except Exception:
                    iv = 0
                if iv > 0 and len(flow_ids) > 0:
                    valid = True
                    rates_to_store = iv

            if not valid:
                if self.debug_logger is not None:
                    try:
                        self.debug_logger.event(
                            "outer_skip_empty_regulation",
                            {
                                "control_volume_id": ctrl,
                                "window_bins": [int(wb[0]), int(wb[1])],
                                "mode": mode,
                                "flow_ids": list(flow_ids),
                                "committed_rates": rates,
                                "reason": "no_effective_rates_or_no_flows",
                            },
                        )
                    except Exception:
                        pass
                # Do not append an empty/no-op regulation; continue to next iteration
                continue

            regulation = RegulationSpec(
                control_volume_id=ctrl,
                window_bins=wb,
                flow_ids=flow_ids,
                mode="per_flow" if mode == "per_flow" else "blanket",
                committed_rates=rates_to_store,
                diagnostics=dict(commit_action.diagnostics or {}),
            )
            state.plan.append(regulation)
            commits += 1
```

```66:101:src/parrhesia/flow_agent/plan_export.py
    plan_items: List[Dict[str, Any]] = []
    for reg in getattr(state, "plan", []) or []:
        try:
            t0 = _safe_int(reg.window_bins[0])
            t1 = _safe_int(reg.window_bins[1])
        except Exception:
            t0, t1 = 0, 1

        flow_ids = list(getattr(reg, "flow_ids", ()) or [])
        mode = str(getattr(reg, "mode", "per_flow"))
        rates = getattr(reg, "committed_rates", None)

        # Skip regulations with no flows or no effective rates
        valid = False
        rates_per_flow_out: Optional[Dict[str, int]] = None
        blanket_rate_out: Optional[int] = None
        if isinstance(rates, dict):
            cleaned = {str(k): _safe_int(v) for k, v in (rates or {}).items() if _safe_int(v) > 0}
            if cleaned and flow_ids:
                valid = True
                rates_per_flow_out = cleaned
                blanket_rate_out = None
        else:
            br = _safe_int(rates) if rates is not None else 0
            if br > 0 and flow_ids:
                valid = True
                rates_per_flow_out = None
                blanket_rate_out = br
        if not valid:
            continue

        item: Dict[str, Any] = {
            "control_volume_id": str(getattr(reg, "control_volume_id", "")),
            "window_bins": [t0, t1],
            # Labels are informational; end label shown for the last included bin (t1-1)
            "window_labels": {
                "start": _bin_label(indexer, t0),
                "end": _bin_label(indexer, max(0, t1 - 1)),
            },
            "mode": mode,
            "flow_ids": flow_ids,
        }
        if rates_per_flow_out is not None:
            item["rates_per_flow"] = rates_per_flow_out
            item["blanket_rate"] = None
        else:
            item["rates_per_flow"] = None
            item["blanket_rate"] = blanket_rate_out

        plan_items.append(item)
```

What this changes
- Commit-time: commits with no positive rates are treated as no-op; the hotspot is closed, but nothing is appended to the plan.
- Agent append: even if a no-op commit slipped through, it’s filtered and not added to `state.plan`.
- Export: any residual empties are skipped from the saved plan.

Next steps
- Run your usual tests and rerun the agent to confirm the exported `plan_*.json` no longer contains entries with empty `flow_ids` or missing/zero rates.

- If you want to also hard-enforce at the data type layer, we can add stricter invariants to `RegulationSpec.__post_init__` (reject empty `flow_ids` or non-positive rates). I left it permissive to avoid disrupting MCTS simulations.

- Optional: log metrics for skipped commits so you can monitor how often the search proposes no-op regulations.

- If you want blank “infinite” rates to be allowed as explicit “no regulation”, we can handle that separately, but they won’t be serialized into plans anymore.

- If you’d like, I can add unit tests that assert:
  - Committing per-flow with all infinite rates does not append to the plan.
  - Export omits invalid regulations.

Summary
- Added commit-time sanitization and skip of no-op regulations in `CheapTransition`.
- Filtered agent-side append to ignore empty/no-rate regulations.
- Plan export now omits any residual empty/no-rate regulations.