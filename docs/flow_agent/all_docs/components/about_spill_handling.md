### What the 87 rate actually means inside a 45‑min window
- **Hourly rate → per-bin quota**: The rate is applied per time bin as quota = round(rate × bin_minutes / 60). With 15‑min bins, 87/h → round(87 × 15/60) = 22 per bin.
- **Window length**: Your window [41,44) spans 3 bins = 45 minutes. So the maximum in-window releases are roughly 22 × 3 = 66, but the first bin only had 21 ready entrants, so total in-window releases = 21 + 22 + 22 = 65.
- **Entrants vs releases**: There were 72 entrants in the window but only 65 releases allowed; the remaining 7 were deferred beyond the window. That matches the log fields: in_window_releases = 65 and spill_T = 7.

Code reference for how the per-bin schedule is built from an hourly rate (note the round-to-integer quota and cumulative “ready minus released” logic):
```649:665:/mnt/d/project-tailwind/src/parrhesia/flow_agent/rate_finder.py
quota = max(0, int(round(rate * bin_minutes / 60.0)))
ready = 0
released = 0
for t in sorted(active_set):
    ...
    ready += demand_list[t]
    available = max(0, ready - released)
    take = min(quota, available) if quota > 0 else 0
    schedule[t] = take
    released += take
...
schedule[T] = overflow_base + max(0, total_non_overflow - scheduled_non_overflow)
```

### Why nonzero delays exist and why the minimum is 22 minutes
- **FIFO within a flow**: Flights are released in requested-time order. If capacity in bins 41–43 is exhausted, remaining flights are pushed to bins after the window.
- **Safe spill mechanics**: After the last active in-window bin, remaining flights are assigned one per successive bin (44, 45, 46, …). Delay is computed from the realised start (start of assigned bin, or the flight’s own requested time if later) minus the requested time; when requested datetimes are known, delays are not locked to 15‑minute multiples.
- **22‑minute minimum**: The first flight that actually incurred positive delay (among those with precise requested datetimes) ended up two bins later than its request and had an 8‑minute within-bin offset, giving 30 − 8 = 22 minutes. Earlier “spilled” flights may have had zero delay (e.g., their requested time was already after the start of the next bin) or lacked precise datetimes (bin-only info yields multiples of 15), so they don’t show up as small values like 2, 5, 7.

Code reference for FIFO delay computation and safe spill:
```323:385:/mnt/d/project-tailwind/src/parrhesia/fcfs/flowful_safespill.py
for _ in range(capacity):
    fid, r_dt, r_bin = ready.popleft()
    if isinstance(r_dt, datetime):
        start_of_bin_dt = _start_of_bin_for_date(r_dt, t, bin_minutes)
        s_dt = r_dt if r_dt >= start_of_bin_dt else start_of_bin_dt
        delay_seconds = max(0.0, (s_dt - r_dt).total_seconds())
        delay_minutes = int(ceil(delay_seconds / 60.0)) if delay_seconds > 0 else 0
...
# Safe-spill after last active in-window bin
spill_start = int(last_active_bin + 1)
k = 0
for fid, r_dt, r_bin in remaining:
    assigned_bin = spill_start + k
    if isinstance(r_dt, datetime):
        start_of_spill_bin = _start_of_bin_for_date(r_dt, assigned_bin, bin_minutes)
        s_dt = r_dt if r_dt >= start_of_spill_bin else start_of_spill_bin
        delay_seconds = max(0.0, (s_dt - r_dt).total_seconds())
        delay_minutes = int(ceil(delay_seconds / 60.0)) if delay_seconds > 0 else 0
    else:
        delay_minutes = max(0, (int(assigned_bin) - int(r_bin)) * bin_minutes)
    k += 1
```

### Is there an “overflowing” problem?
- **No**. The schedule `n` includes an overflow count at index T for diagnostics, but the actual realisation uses safe spill into bins after the last active in-window bin. In your run, spill_T = 7 simply means 7 flights were not released inside [41,44) and were scheduled into bins 44+; it’s expected given 72 entrants vs ~65 in-window capacity.

If you wanted zero delays inside that 45‑minute window, you’d need the hourly rate high enough that 3×quota ≥ 72, i.e., rate ≥ 96/h for 15‑min bins, or use a longer window.

- The observed numbers are internally consistent with the code and the log:
  - per-bin quota = 22, releases in [41,42,43] = 21,22,22 ⇒ 65
  - entrants in window = 72 ⇒ 7 spilled
  - nonzero delays (e.g., 22, 35, 50, …) arise from multi-bin deferrals minus within-bin offsets.

Summary:
- 87/h over a 45‑min window allows at most about 65 releases; 7 of 72 entrants must spill, so delays are expected.
- Delays are from FIFO + safe spill and within-bin timing; 22 minutes minimum is consistent with a two-bin deferral minus an 8‑minute offset.
- No overflow bug; behavior matches the implemented scheduling and diagnostics.

### How “spilling” works
- **In-window uses the chosen rate**: Inside the regulation window, releases are capped by the per‑bin quota derived from the hourly rate and FIFO within each flow.

- **After the window, rate is NOT preserved**: Remaining flights are “safe‑spilled” one‑by‑one into consecutive bins starting right after the last active in‑window bin. This is a deterministic drain of the backlog; it does not try to maintain the in‑window hourly rate and does not consult capacities outside the window.
  - Practically: if 7 flights remain and the window ends at bin 43, they are placed at bins 44, 45, 46, … (one per bin). With 15‑minute bins, that yields roughly 15‑minute spacing of spilled flights.
  - If a flight’s requested time is later than the assigned spill bin’s start, its realised start snaps to its requested time (no artificial early release), so some spilled flights can still have 0 delay.

- **Overflow count is diagnostic only**: The schedule’s overflow entry `n[T]` captures “not released in window,” but actual realisation ignores that bucket and uses the safe‑spill ladder. A warning is logged if `n[T]` disagrees with the true remainder, but safe‑spill is still used.

Code references
```346:385:/mnt/d/project-tailwind/src/parrhesia/fcfs/flowful_safespill.py
# Find last active in-window bin with capacity, then safe-spill
last_active_bin = -1
for tt in range(T - 1, -1, -1):
    if int(schedule[tt]) > 0:
        last_active_bin = int(tt)
        break
...
overflow_cap = int(schedule[T]) if T < len(schedule) else 0
if overflow_cap != len(remaining):
    logging.warning(
        "Flow %s: overflow count %d != remaining %d; using safe-spill distribution",
        str(flow_id), overflow_cap, len(remaining),
    )

# Assign remaining flights one per successive spill bin (rate not preserved)
spill_start = int(last_active_bin + 1)
k = 0
for fid, r_dt, r_bin in remaining:
    assigned_bin = spill_start + k
    if isinstance(r_dt, datetime):
        start_of_spill_bin = _start_of_bin_for_date(r_dt, assigned_bin, bin_minutes)
        s_dt = r_dt if r_dt >= start_of_spill_bin else start_of_spill_bin
        delay_seconds = max(0.0, (s_dt - r_dt).total_seconds())
        delay_minutes = int(ceil(delay_seconds / 60.0)) if delay_seconds > 0 else 0
        realised_start[fid] = s_dt
        delays_min[fid] = delay_minutes
    else:
        delay_minutes = max(0, (int(assigned_bin) - int(r_bin)) * bin_minutes)
        realised_start[fid] = {"bin": int(assigned_bin)}
        delays_min[fid] = int(delay_minutes)
    k += 1
```

- **Takeaway**: **Spilling is a one‑per‑bin drain after the window**, not a continuation of the hourly rate and not a dump into a single next bin. It’s designed to avoid end‑of‑day teleport and keep delays minimal/monotone under FIFO.

- In your case: 7 remainders were placed into consecutive bins after 43; within‑bin offsets explain why the smallest positive delay you saw was 22 (not necessarily 15).

### Original `flowful.py` overspill behavior
- **Inside window**: FIFO with per-bin capacity from the chosen hourly rate (same as now).
- **After the window (overspill)**:
  - Uses a single, hard overflow bin at index `T` (end-of-day).
  - Releases up to `n_f(T)` remaining flights into bin `T`. It does not preserve the in-window rate and does not spread flights across `T+1, T+2, ...`.
  - If `n_f(T)` is less than the number of remaining flights, it truncates and logs a warning; some flights may remain unscheduled (infeasible schedule).
  - Delays are computed to the start of bin `T` (unless the flight’s requested time is after that, in which case delay can be 0).

Code reference:
```339:370:/mnt/d/project-tailwind/src/parrhesia/fcfs/flowful.py
# Overflow bin T: release remaining flights
remaining: List[Tuple[str, Optional[datetime], int]] = list(ready)
if i < N:
    remaining.extend(flights_norm[i:])

overflow_cap = int(schedule[T]) if T < len(schedule) else 0
if overflow_cap != len(remaining):
    if overflow_cap < len(remaining):
        logging.warning(
            "Flow %s: overflow capacity %d < remaining %d; truncating",
            str(flow_id), overflow_cap, len(remaining),
        )
    else:
        logging.warning(
            "Flow %s: overflow capacity %d > remaining %d; extra capacity unused",
            str(flow_id), overflow_cap, len(remaining),
        )
to_release = remaining[: max(0, min(overflow_cap, len(remaining)))]

for fid, r_dt, r_bin in to_release:
    if isinstance(r_dt, datetime):
        start_of_overflow = _start_of_bin_for_date(r_dt, T, bin_minutes)
        s_dt = r_dt if r_dt >= start_of_overflow else start_of_overflow
        ...
    else:
        delay_minutes = max(0, (T - int(r_bin)) * bin_minutes)
        realised_start[fid] = {"bin": int(T)}
        delays_min[fid] = int(delay_minutes)
```

- This “dump to T” is why the safe-spill variant was introduced: to avoid end-of-day teleport, keep feasibility without relying on `n[T]`, and distribute leftovers across successive bins after the window.
