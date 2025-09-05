# Goals

In the context of the current codebase, I would like you to plan in detail so that the Simulated Annealing (SA) will plan for the **release rate**, while the FCFS delay assignment algorithm will grant delay to the flights so as to respect the target rates set by SA. 

The important thing is that in the current implementation, we store the demand time bin, but not the concrete datetime which makes it impossible for the FCFS to assign a delay lower than the time-bin length of 15 minute. However, I believe that the traffic volume entry time (which is the precursor for the missing information) is already available in the FlightList (the `so6_occupancy_matrix_with_times.json`). Here is a small excerpt:

```json
"263854795": {
    "occupancy_intervals": [
        {
            "tvtw_index": 33504,
            "entry_time_s": 0.0,
            "exit_time_s": 38.0
        },
        {
            "tvtw_index": 33408,
            "entry_time_s": 0.0,
            "exit_time_s": 38.0
        },...
```

The entry_time_s is second since takeoff. 

> Please give a detailed plan (hold off writing any code yet at this moment) to realize the goal of relying on the `entry_time_s` (and probably `exit_time_s` if you need to, but only if you need to, because I don't know at this moment whether you will need it), for the FCFS delay assigner to be able to properly assign the delay (not just a 15-minute multiplicity anymore).

# Attention

There are also some legacy code that does queueing/slot allocation located in the `casa` module. Please disregard it. Use `fcfs` module's (`flowful` and `scheduler` for instance).

# More Context

The SA endpoint only schedules at bin granularity, and your bin size is 15 minutes. Because it feeds the scheduler only “which bin” each flight requested (not the exact minute), the FIFO assignment computes delays in whole-bin increments, i.e., multiples of 15 minutes.

Why it happens

Bin-only inputs: The SA prep builds flight specs with only requested_bin, no requested_dt per flight.
src/parrhesia/optim/sa_optimizer.py
src/parrhesia/api/automatic_rate_adjustment.py
FIFO delay logic: When a flight spec lacks a datetime, delay is computed as whole bins:
delay_minutes = (realized_bin − requested_bin) × time_bin_minutes
src/parrhesia/fcfs/flowful.py
Explicit note in code docs: “when only bin indices are available, delays are computed as multiples of the bin length.”
src/parrhesia/fcfs/flowful.py
Where delays are produced for the API: SA scoring calls the FIFO to produce delays_min, which your endpoint returns.
src/parrhesia/optim/objective.py
What this means

With time_bin_minutes = 15, all delays_min are multiples of 15.
The example in API_README_SA.md can show 5-minute delays when true datetimes are used; your current SA path does not pass those.
How to get sub-15-minute delays (optional)

Provide within-bin times to the scheduler by including requested_dt, or the pair takeoff_time + entry_time_s, in each flight spec used by SA (so _normalize_flight_spec can recover a datetime).
See accepted formats in src/parrhesia/fcfs/flowful.py
–45
Concretely: in prepare_flow_scheduling_inputs, for each flight at the chosen controlled volume, set requested_dt using flight metadata (e.g., takeoff_time + the interval’s entry_time_s for that TV). Then delays will be minute-precise (ceil of seconds/60) rather than bin-sized. This is now implemented: SA enriches flight specs with `requested_dt` while preserving `requested_bin` for demand computations.

1) What SA Optimizes

Rate per flow per bin: SA’s decision variable is n_f(t), an integer release count for each flow f and time bin t (plus overflow at T). It does not assign flights to bins directly.

n_f(t) is initialized from demand counts and then modified by SA moves; see building n_by_flow from demand counts: src/parrhesia/optim/sa_optimizer.py
Per-flight specs used by SA contain only a requested_bin (not a specific per-flight assignment): src/parrhesia/optim/sa_optimizer.py
The API builds those specs via controlled-volume selection, then SA optimizes the counts: src/parrhesia/api/automatic_rate_adjustment.py
Candidates are evaluated by scoring with the FIFO scheduler; SA never “pins” a specific flight to a specific bin: src/parrhesia/optim/sa_optimizer.py
Not per-TV cell: The variable is per flow/time-bin at the controlled volume. Occupancy across TVs is evaluated after scheduling using flight footprints; SA does not set per-TV rates directly.

2) Rate → Delay Path

Build specs: For each flow, the API/SA prep constructs flights_by_flow with each flight’s requested_bin at the chosen controlled volume; no per-flight requested_dt is provided in this path: src/parrhesia/optim/sa_optimizer.py

SA output: SA proposes n_opt — per-flow counts per bin (length T+1): src/parrhesia/optim/sa_optimizer.py

Schedule to delays (FIFO):

Scoring calls the FIFO scheduler with the pre-sorted flights and n_opt to get per-flight delays: src/parrhesia/optim/objective.py
FIFO logic per bin t: enqueue flights with r_i ready by end of t; release up to n_f(t) in FIFO order; for each released flight i:
If a datetime exists: s_i = max(r_i, start_of_bin(t)), delay = ceil((s_i − r_i)/60s) minutes: src/parrhesia/fcfs/flowful.py
For bin-only specs (this SA path): delay = (t − requested_bin) * time_bin_minutes minutes (whole-bin increments): src/parrhesia/fcfs/flowful.py
Overflow bin T releases any remaining flights with the same rules: src/parrhesia/fcfs/flowful.py
Returned delays: The artifacts from scoring include delays_min (per-flight minutes), which the endpoint returns. This is why you observe delays in multiples of the bin size when only bin indices are provided (explicitly noted here): src/parrhesia/fcfs/flowful.py

Net: SA sets the per-flow release rates per time bin; the FIFO scheduler maps those rates to individual flights and computes per-flight delays.
