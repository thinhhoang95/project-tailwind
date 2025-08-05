# Module Name
> **Nominal count and regulation cost** (short name: `nominal`), as part of the evaluation function computation.

# High level inputs - outputs
## Inputs
1. Original flight count (as a vector of `N` traffic-volume-time-windows (TVTWs)). This is a huge sparse matrix, called the *occupancy matrix*, one row for each flight, and one column for each TVTW. The indexing of TVTW follows the `tvtw_indexer.py` module.
2. The network plan, which is a set of *regulations* to be implemented (roughly speaking, a regulation is a list of flights to be delayed through setting the entry rate at some traffic volumes or location).
3. The horizon H (in time windows).

## Outputs
1. The nominally updated flight count as the regulations are implemented.
2. The total excess traffic (within the horizon).
3. The maximum excess traffic in any of the (within the horizon).

# Implementation Details
### How regulations work
Each *network plan* is a set of *regulations*. Each *regulation* is a triplet of `<REFERENCE LOCATION OR TRAFFIC VOLUME> <FILTERING CONDITION> <RATE> <TIME WINDOWS>`. 

### General Approach
1. From the original flight plans (for example: from `D:/project-cirrus/cases/flights_20230801.csv`), we compute the *TVTW occupancy vector*, which is one row in the *occupancy matrix*. This calculation needs to be done once only.

2. For each *regulation* in the *network plan*, we look for the set of flights that are eligible to be targetted with delay.

    Comprehensively, we use the `<REFERENCE LOCATION>` and `<FILTERING CONDITION>` to select the flights. This is a simple matching problem. 

    For example: `<LFPPLW1> <LFP* > LI*> <60> <36, 37>` means that we select the flights that cross through the traffic volume `LFPPLW1`, going from any Paris airports e.g., LFPG, LFPO; to any Italian airports e.g., LIMC; and the crossing time is in time window 36, 37 (9:00-9:30 AM) for a time window of 15 minutes. 

    In this evaluation function implementation, we will assume the regulation is given (how we arrive at it is not relevant at this point).

3. After selecting the flights, we run a variant of the *Computer Assisted Slot Allocation* (more precisely called C-CASA) on this set of flights, with the given rate, and the C-CASA machinery spits out the ground delay value for each flight (in minutes).

4. Now each flight assigned with the delays, they might either:

    a. Accept the delay, if the delay is less than 25 minutes (this is a heuristic at this moment and will be refined eventually with more complex logic, but at this moment, we are content with this heuristic).

    b. If the delay is more than 25 minutes, it will seek for a reroute to dodge the delay. This is in fact can be done through a placeholder function call for our preliminary implementation. There are two outcomes: if a rerouting solution is found, a **TVTW occupancy vector** corresponding to the rerouting solution is returned. If the rerouting solution is not found, it will accept the original delay.

5. If the flight was rerouted, the *TVTW occupancy vector* is taken directly. If the flight was delayed, we just "shift the time bin indices" for the *TVTW occupancy vector* according to the delay in minutes, then we replace the original *TVTW occupancy vector* of that flight with the "shifted" one.

**Remarks:** technically there is still a possibility that a rerouted flight might still need to be delayed. However, this will complicate our calculations because the C-CASA algorithm needs to be invoked again. As a result, we propose a first order assumption that the rerouted flight will not induce any secondary delay. This will be compensated (if necessary) in the next cycle of the control.

6. The rerouting function will also return the cost difference (additional fuel burn, crew time...). If the flight was rerouted, and there was no additional delay, 

6. We then find out the TVTW with most excess traffic (i.e., the count subtracted by capacity), during the horizon H (i.e., considering all time windows within H). This will be `z_max`. We also compute `z_sum` by summing all excess traffic across all TVTWs within the horizon H.

7. We return the new occupancy matrix, `z_max` and `z_sum`.

# Specific Instructions
The current task deals **exclusively** with the **re-adaptation of the current C-CASA code** so that it works with the current data format (for flight plans, and regulation).

#### Inputs
1. The original flight list is loaded from `output/so6_occupancy_matrix_with_times.json`. An excerpt of the file is as follows:

    ```json
    {
        "263854795": { % flight_identifier
            "occupancy_intervals": [
                {
                    "tvtw_index": 13920,
                    "entry_time_s": 0.0,
                    "exit_time_s": 307.0
                },
                {
                    "tvtw_index": 12192,
                    "entry_time_s": 241.0,
                    "exit_time_s": 751.0
                },
                {
                    "tvtw_index": 13728,
                    "entry_time_s": 241.0,
                    "exit_time_s": 848.5
                }
            ],
            "distance": 104.14351906299255,
            "takeoff_time": "2023-08-01T00:01:15",
            "origin": "LBBG",
            "destination": "EGFF"
        },...
    }
    ```

    Here the entry and exit time are measured in seconds since takeoff, and the `tvtw_index` follows the convention in `tvtw_indexer.py` module.

2. The `identifier_list` which is a list of flight_identifiers (as string) parsed from the input network plan (which is nothing more than just a list of regulations) to be targetted.

3. The `reference_location`, which can either be the list of waypoints, or the traffic volume ID. We only suppor the traffic volume ID for now.

4. The `tvtw_indexer` object, which helps bookkeeping the conversion between traffic volume time windows, IDs and their corresponding indices.

5. The `hourly_rate` value, which is a float.

6. The `active_time_windows`, a list of time window indices, indicating the period when the regulation is **active**. For example: `[46, 47, 48, 49]`.

#### Outputs
We return a dictionary which assigns the delay in minutes to each flight in the `identifier_list`.

#### Details

1. Given the `reference_location` (which is in our case, the `traffic_volume_id`), and the `active_time_windows`, we find a set of TVTW indices from the `tvtw_indexer`. We call this set the `eligible_tvtw_indices`.

2. The next step is to start the C-CASA algorithm from the beginning of the first eligible time window (for example: 46). It will start with the first C-CASA's `ccasa_window_length_min` (min = minutes; note that this is a distinct from the existing time windows/bins from the input data), ensure that the rate does not exceed the `hourly_rate` divided by `60/ccasa_window_length_min`. It will keep the earliest flight arriving in that window, and push the "excessive flights" to the end of the window. Then it will slide the window to the right an amount of `window_stride_min` minutes, and start again. The process will stop when the beginning of the window fell out of the `eligible_time_window_indices`.

    The tricky part is that the given input will show you the entry time at each TVTW, but each time window there might have a length that is smaller, equal or greater than the C-CASA's window_length_min. 

#### Specific Instructions
Below is a **robust, C-CASA-compatible plan** to re-adapt the algorithm to your current data format and regulation inputs. It deliberately **does not use a headway model**; it treats CASA as a **rate-limiting queue at a point** (the reference traffic volume), counting **entries** in rolling windows and pushing excess entries to later windows.

---

## 0) Timescales and primitives (align definitions up-front)

* **TVTW bins** (e.g., 15- or 30-min): the discretization used to build the *occupancy matrix* and to define regulation *activity windows*.

* **CASA rolling windows**: independent analysis windows of length `ccasa_window_length_min` that slide by `window_stride_min`.

* **Hourly rate → per-CASA-window capacity**:

  $$
  C_\text{win}=\left\lfloor \text{hourly\_rate}\times\frac{\text{ccasa\_window\_length\_min}}{60}\right\rfloor
  $$

  plus a *fractional capacity accumulator* (see §3.d) so fractional rates (e.g., 22/h with 5-min windows) are honored without bias.
  (This mirrors how the previous code derived a window threshold from an hourly capacity schedule, but here we use your single `hourly_rate` parameter. )

* **Counting rule**: CASA counts **entries** at the reference location in $[W_\text{start}, W_\text{end})$; it does **not** depend on the duration of occupancy through that TVTW. This entirely sidesteps any mismatch between TVTW bin size and CASA window length.

---

## 1) Build the **eligible set** and derive **absolute entry times**

**Inputs used:**

* `identifier_list` (target flights),
* `reference_location` (traffic volume ID),
* `active_time_windows` (list of TVTW **time indices**),
* `tvtw_indexer` (mapping TV/time → global index),
* `output/so6_occupancy_matrix_with_times.json` (per-flight occupancy intervals with `tvtw_index`, `entry_time_s`, `exit_time_s`, `takeoff_time`).

**Steps**

1. **Compute the set of eligible TVTW indices**

   $$
   E=\{\texttt{get\_tvtw\_index(reference\_location, tw)}\ :\ tw\in \texttt{active\_time\_windows}\}
   $$

   using the `TVTWIndexer` already provided.&#x20;

2. **Parse each target flight** from JSON. For each `occupancy_intervals` element:

   * Map `tvtw_index → (tv_id, time_window_idx)` with `get_tvtw_from_index`.&#x20;
   * If `tv_id == reference_location`, compute **absolute** entry/exit instants:
     `t_entry_abs = takeoff_time + entry_time_s`,
     `t_exit_abs  = takeoff_time + exit_time_s`.
     (We prefer these real times over bin-implied times to avoid edge drift.)

3. **Eligibility check — robust to bin/edge effects**
   Include a crossing if **either**:

   * Its global `tvtw_index ∈ E` (direct match), **or**
   * `tv_id == reference_location` **and** `floor(minute_of_day(t_entry_abs)/time_bin_minutes) ∈ active_time_windows`.
     This second clause ensures we retain true crossings that fall near a bin boundary even if the precomputed `tvtw_index` doesn’t land exactly in `E`.

4. **If a flight crosses multiple eligible bins** of the same `reference_location` (rare but possible due to pathologies/holds), keep the **first** `t_entry_abs` within the active period.

5. **Create a CASA event** per eligible flight:
   `(flight_id, t_entry_abs, tv_id=reference_location)`.
   Sort the event list by `t_entry_abs` (FIFO).

---

## 2) Anchor CASA windows to the **active period**

1. Convert the **active TVTW span** to absolute time:

   * `active_start_abs`: start of the **first** active time index on the day.
   * `active_end_abs`: end of the **last** active time index on the day.
     (`tvtw_indexer.time_window_map` helps produce the human-readable ranges; you map those to day-absolute times for the date of interest. )

2. Generate rolling CASA windows $[W_i^\text{start}, W_i^\text{end})$ from `active_start_abs` until `W_i^\text{start} < active_end_abs`. (This matches the rolling-window orchestrator pattern in your current `casa.py`, but replacing “takeoff-time counting” with **entry-time at TV** counting. )

---

## 3) CASA queueing on entries (rate control without headway)

For each CASA window $i$:

a) **Collect entrants**: all events with current scheduled crossing time $t^\star$ in $[W_i^\text{start}, W_i^\text{end})$. (Each event’s time starts as `t_entry_abs` and may be **revised** by prior pushes.)

b) **Allow up to $C_i$ entries** now, where

$$
C_i=\left\lfloor \text{hourly\_rate}\times\frac{\text{ccasa\_window\_length\_min}}{60}+\xi_i\right\rfloor
$$

and $\xi_i$ is the **fractional accumulator** from §3.d below.

c) **Push excess to the end of the window (FIFO)**:

* Keep the first $C_i$ flights (by $t^\star$).
* For each remaining flight $f$, set its **revised crossing time**
  $t^\star \leftarrow \max\{t^\star,\;W_i^\text{end}+\epsilon\}$, where $\epsilon$ is a small buffer (e.g., 1–5 seconds) to guarantee it falls strictly outside the current window.
* Record a **delay at the reference location**: $\Delta_f = t^\star - t_{\text{entry,orig}}$.

*(This mirrors the intent of your current `assign_delays` helper—which moves excess flights just beyond the window—but adapted from “revised takeoff time” to **revised crossing time at the TV**. )*

d) **Fractional capacity accumulator (important with short windows)**:
When $C_\text{win}$ is fractional (e.g., hourly\_rate=23 and 5-min windows → 1.9167 per window), we avoid bias by carrying the fractional part across windows:

* Maintain `carry += hourly_rate * (window_len/60)`.
* Set $C_i=\lfloor \text{carry}\rfloor$, then `carry -= C_i`.
  This reproduces the long-run hourly throughput exactly, even when window length/stride are small.

e) **Slide to the next window** and repeat until $W_i^\text{start}$ exits the active period.

---

## 4) Behavior at the **end of regulation**

Two reasonable policies (pick one and document it):

* **Strict end**: The last CASA window ends at `active_end_abs`. Any flights still pushed by that window get $t^\star = active_end_abs + \epsilon$. No further rate checks apply.
* **Graceful unwind (often preferable)**: After `active_end_abs`, continue “windows” with **infinite capacity** so delayed flights retain their revised times without additional pushes—i.e., no capacity constraints beyond the regulation horizon.

Either way, the **computed delay per flight** is $\Delta_f$ (minutes) at the reference location. These are the numbers you return in the dictionary for the `identifier_list`. (Your existing pipeline later interprets these as ground delays to shift each flight’s TVTW occupancy vector; that step is outside the scope here.)

---

## 5) Edge cases & robustness

* **Bin mismatch & boundary flights**: The dual eligibility test in §1.3 ensures we don’t drop a crossing due to a rounding mismatch between prebuilt `tvtw_index` and actual `t_entry_abs`.
* **Multiple crossings of the same TV in the active period**: keep the **first** crossing per flight.
* **Flights exactly on window boundaries**: use **left-closed, right-open** intervals $[W_\text{start}, W_\text{end})$ and a positive $\epsilon$ when pushing to avoid ping-pong at edges.
* **Very small or zero occupancy intervals**: irrelevant to CASA—only **entry** time is counted.
* **`hourly_rate < 60 / ccasa_window_length_min`** (i.e., fewer than 1 flight per window on average): the **fractional accumulator** prevents starvation or periodic under-utilization.
* **Missing or malformed records**: skip flights lacking `takeoff_time` or `entry_time_s` for the target TV; log and return zero delay for them.
* **Day boundaries/time zones**: interpret `takeoff_time` in the dataset’s timezone; derive `active_start_abs`/`active_end_abs` for the same day reference. (If flights cross midnight, keep absolute timestamps consistent.)
* **Overlapping regulations** (future work): if two simultaneous regulations target the same flight at different TVs, run CASA per regulation and take the **maximum** resulting delay at the flight level, or implement a combined multi-constraint CASA (out of scope here).

---

## 6) Data structures and outputs

* **CASA event**: `{flight_id, t_entry_orig, t_entry_star, tv_id, eligible=True/False}`
* **Return value**: `dict[str, float]` mapping `flight_id → delay_minutes = (t_entry_star - t_entry_orig) / 60`. For flights in `identifier_list` that never became eligible, return `0.0`.

---

## 7) Notes on integrating with your current codebase

* In your existing `casa.py`, **window generation** and **capacity per window** logic are already present; the main change is to swap the *counting basis* from “revised takeoff times” to **entry events at the reference TV** (and to take `hourly_rate` from the regulation rather than a per-TV capacity schedule).&#x20;
* The current `assign_delays` pushes delayed flights **to just after the window end** relative to a reference flight; the exact same behavior should be applied to **revised crossing times**, not takeoff times. (You’ll keep FIFO by the `t_entry_star` order rather than `revised_takeoff_time`.)&#x20;
* `TVTWIndexer` already provides the mapping utilities you need (`get_tvtw_index`, `get_tvtw_from_index`, `time_window_map`); ensure its `time_bin_minutes` matches the data used to build `occupancy_matrix_with_times.json`.&#x20;

---

## 8) Why this handles “TVTW vs CASA window” robustly

* CASA only needs **entry counts** in rolling windows; it **does not require** window lengths to match occupancy bins.
* By basing everything on **absolute entry times** (`takeoff_time + entry_time_s`) and using the dual eligibility test, you avoid off-by-one and bin-edge artifacts.
* The **fractional capacity accumulator** ensures accurate long-run enforcement of `hourly_rate` even when CASA windows are short or when the stride doesn’t divide 60.
* Pushing excess **to the end of the current CASA window** maintains the intended C-CASA behavior without invoking headway logic.

If you’d like, I can translate this into a short implementation plan diffed against your current `casa.py`/`delay_assigner.py` (no new dependencies) and a unit-test harness that feeds synthetic entry events through the algorithm.  &#x20;
