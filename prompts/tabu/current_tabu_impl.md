### Implemented moves
- **AddFlight**: Add a top-ranked candidate flight to an existing regulation’s targets.
- **RemoveFlight**: Remove one existing target flight from a regulation.
- **AdjustRate**: Increment or decrement a regulation’s rate by ±`rate_step`.

### Simplifications/trade-offs
- **Single global beam**: The `beam_width` caps total proposals; the first move type(s) can fill it, potentially starving later move types in a given iteration.
- **Candidate pool scope**: Candidates are only flights matching a “blanket” regulation for the same `location` and `time_windows`; no exploration across locations or windows.
- **Ranking seed**: `rank_candidates` is called with an empty seed footprint (doesn’t bias similarity to current targets); could be improved by using each regulation’s current footprint.
- **Move set**: No moves for add/remove regulation, change time windows, change location, swap/transfer flights between regulations, or multi-flight batch edits.
- **Tabu policy**: Simple reverse-key tabu with tenure; aspiration allows tabu moves only if they strictly improve the global best.
- **Performance-only change**: I didn’t alter move logic; I added injection of precomputed `NetworkEvaluator` and `FlightFeatures` and updated the test to compute them once and reuse them.

If you want, I can add per-move beam quotas, seed-aware ranking, and new moves (e.g., AddRegulation, RemoveRegulation, ShiftWindow, SwapFlight) next.