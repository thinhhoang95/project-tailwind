Short answer: the search is bouncing between two already-expanded nodes because of how the PUCT scores line up, and because visits aren’t being incremented (or otherwise penalized) during this in-simulation loop. Nothing changes the scores, so it keeps picking the same two actions forever.

What the log shows

You alternate between two nodes:
select_flows node 1cb90b7654b7, where the chosen action is always cont
confirm node b61ca92ecac6, where the chosen action is always back
At select_flows:
cont has an enormous Q = 9269.5, while the unvisited rem(…) moves have Q = 0 and exploration U ≈ 0.34.
Score = Q + U, so cont’s score ≈ 9269.69 dwarfs every alternative (≈ 0.34). You never try rem(…).
At confirm:
commit has been tried once (N_edge=1, Q=0), so its exploration term is smaller: U ≈ 0.4083.
back and stop are unvisited, so their exploration terms are larger: back U ≈ 0.636, stop U ≈ 0.547.
With Q = 0 for all three, back always wins on U, so you go “back” to select_flows again.
Why this turns into a loop

The transition cont → back → cont returns you to the same select_flows state (z_hash stays the same; phi_s == phi_sp; reward = 0). It’s a pure loop with no progress.
Critically, the node and edge visit counts don’t change inside this loop:
confirm node shows node_N = 1 every time, and N_edge(back) stays 0, even though you keep selecting back.
select_flows node shows node_N = 2 and keeps cont’s edge stats unchanged.
Because N and N_edge don’t increase during this intra-simulation loop, the PUCT exploration bonuses (U) don’t decay, and the ordering never flips. back keeps beating commit/stop at confirm, and cont keeps beating rem(…) at select_flows.
Likely underlying issues

Scale mismatch: Q(cont) is huge (9269.5) relative to U (~0.2–0.6). That virtually guarantees cont wins at select_flows, blocking exploration of rem(…). Note 9269.5 equals roughly -phi/2 given phi_s = -18539; if that’s intentional, it still needs normalization so Q is commensurate with U.
No within-simulation visit updates or cycle handling: because you don’t increment N/N_edge (or apply virtual visits/loss) as you traverse, the U terms for back/stop don’t change, so you never break the tie by experience. And there’s no cycle detection to prevent returning to a node already on the current path.
How to prevent getting stuck

Normalize/scale Q so it’s on the same order as the U term, or increase c_puct accordingly; typical MCTS keeps Q in a bounded range (e.g., [-1, 1]).
Increment edge/node visits (or apply “virtual visits/loss”) immediately on selection, not only at backup, so repeated choices in the same simulation reduce their U and allow alternatives to surface.
Add cycle detection for a single simulation path (e.g., if selecting an action would revisit a node already on the path, disallow it, penalize it, or treat it as terminal with a small negative).
Consider domain rules to avoid immediate backtracking (e.g., forbid back right after cont), or add a small step penalty to break indifference.
If appropriate, inject exploration (e.g., Dirichlet noise) at the relevant node to let rem(…) get tried at least once.
In short: the agent “gets stuck” because it enters a deterministic cont ↔ back loop where PUCT scores never change (no visit updates, huge Q on cont), so the same two choices keep winning and the simulation never progresses to expansion or backup.