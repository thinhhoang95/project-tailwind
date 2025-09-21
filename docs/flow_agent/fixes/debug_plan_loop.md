### Next steps to hunt it down

- Verify state key stability
  - Log a short hash of `state.canonical_key()` on every visit, plus `stage`, and whether the node existed or was created. Confirm the confirm-state key is identical across `select_flows → Continue → confirm → Back → select_flows → Continue → confirm`.
  - Also log a short hash of `z_hat.tobytes()` to ensure no drift across `Continue/Back`.

- Check node reuse and edge stats across sims
  - On every visit to confirm, print: `node.N`, `len(node.children)`, and for each of `commit/back/stop`: `P`, `N_edge`, `Q`.
  - After `_backup`, print updated `N_edge/Q` for the edge just taken so you can see if they actually accumulate.

- Decompose the selection score
  - In the confirm state logs, print both parts explicitly: `score = Q + (c_puct * P * sqrt(N_parent) / (1+N_edge))`. Verify numbers match and whether ties happen; remember lexicographic tie-break favors `back` over `commit`.

- Inspect reward shaping and leaf bootstrap
  - For steps around the loop, print `phi(s)` and `phi(s')` so you can see `Δphi`. `Continue` and `Back` should have `Δphi=0`. Confirm the leaf bootstrap at first-time visits uses `-phi(leaf)` as intended and is what’s inflating upstream Q.

- Confirm priors at confirm
  - Log the computed priors at confirm; they should come out with `commit > back > stop`. If they’re equal/near-zero, debug why (temperature, logits, unexpected candidates).

- Audit backprop path integrity
  - Log the full `path` as `(node_hash, action_sig)` just before `_backup`, and then the same nodes’ `N/Q` after, to ensure the intended nodes/edges are being updated.

- Check commit evaluation budget
  - In your runner you set `commit_eval_limit=-1`. With current code, that makes every commit return `delta_j=0` (treated as budget exhausted). Temporarily run with a positive value (e.g., 16) and compare behavior; this can strongly affect Q at confirm.

- Minimal repro sim
  - Run a tiny scenario (one hotspot, a few flows, fixed proxies) with `debug_prints=True`. Stop after ~50 sims. You should see: confirm-node hash stable, `N_edge/Q` for confirm actions increasing across sims, and selection occasionally choosing `commit`.

- If the loop persists with stable keys
  - Print `U` components at select_flows vs confirm to see if the continue edge’s large Q (from leaf bootstrap) starves confirm of visits. If so, consider temporarily lowering `c_puct` or increasing priors for `commit` to validate the hypothesis without changing code permanently.

- Sanity checks
  - Ensure `selected_flow_ids` are unchanged across `Back/Continue`.
  - Verify no NaNs/Infs in `z_hat` or priors; add guards to log and skip if detected.

- Optional one-off diagnostics
  - For confirm only, log whether the visited node was created this sim vs pre-existing in `self.nodes` and print the first 12 chars of its key to correlate across sims.

- Seed/replication
  - Run with a fixed seed and once with a different seed (even though selection is deterministic) to confirm determinism of the loop.

- Data capture for offline inspection
  - Dump a small JSON trace per sim with: stage, node_hash, children signatures with (P,N,Q), chosen action, `phi`, `Δphi`, total return. Diff traces between “looping” vs “non-looping” runs.

- If keys are unstable after all
  - Narrow it to which field flips: compare two consecutive confirm payloads (plan, context, stage, z_hat hash). Focus on `z_hat` content and `candidate_flow_ids` order.

- If keys are stable but `back` dominates
  - You’ve confirmed a Q update dynamic: continue’s edge gets repeatedly boosted by bootstrap, confirm edges stay cold. Validate by artificially warming `commit` (e.g., 1 forced rollout per visit) to see if loop breaks; that isolates Q/backprop as the root cause.

- Final quick check in code reading
  - Ensure `_signature_for_action` for `commit` is only based on chosen flows (it is) and not on `committed_rates`, so edges aggregate correctly.

- Highest-impact checks first
  - 1) commit_eval_limit positive vs -1
  - 2) confirm-node key/hash stability across the loop
  - 3) per-edge `N/Q` at confirm increasing across sims (or not)

- What to collect to decide next fix
  - A 30–50 sim trace showing confirm-node key stability, confirm-edge `N/Q` across time, and selection scores; with that we can tell if it’s key instability or Q/backprop dynamics.