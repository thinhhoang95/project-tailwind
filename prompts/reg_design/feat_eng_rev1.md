# Goal

To revise the price kernel vG and the contribution in the matched-filter score with the new set of improved formulas.

# Details

- mass weight gH(x) = x/(x + τH) with τH = DH, and
- overload-addressable share ωs,G|H.

Setup for hotspot H = (s*, t*)
- Phase: tG = t* − τG,s*. -> this is the time at control volume, which the traffic when released at tG will arrive at the hotspot TV s* at t*.
- Reference mass at phase: x̂GH = xG(tG). -> this is just the flow's count at the control volume at the phase time given above.
- Hotspot exceedance: DH = [os*(t*)]+. -> this is the number of flights "exceeding" the capacity values. Remember: all capacity values are comparable to **rolling-hour** occupancy count: at each moment, for the next 60 minutes.

Mass weight (replaces eligibility gate)
- gH(x̂GH) = x̂GH / (x̂GH + DH). -> this is a materiality/coverage term. It says: “how big is this flow relative to what we need at H?” When DH is large, any single small flow covers a small fraction of the need, so its materiality should be down-weighted—while larger flows still rank higher because gH is monotone in x̂GH.

Sectorwise alignment for G relative to H
For each sector s touched by G:
- Phase-aligned time: ts = t* + τG,s − τG,s*. -> literally, the corresponding time at s when the time at hotspot is t*.
- Cell overload: Ds = [os(ts)]+. -> the exceedance at ts. 
- Price at cell: Ps = (wsum + wmax θs,ts) 1{os(ts) > 0}.
- Flow presence at cell (reference): xs,G = xG(s, ts) under the unregulated plan.
- Overload-addressable share: ωs,G|H = min{1, xs,G / (Ds + ε)}, with small ε > 0.

Contribution-weighted cell-specific unit price
- ṽG→H = Ps* ωs*,G|H + κ ∑s≠s* Ps ωs,G|H, with κ ∈ [0, 1].

Slack-risk penalty (unchanged)
- SlackG(t) = mins ss(t + τG,s), so SlackG(tG) = mins ss(t* + τG,s − τG,s*).
- ρG→H = max{0, 1 − SlackG(tG) / S0}.

Matched-filter net score
- Score(G | H) = α gH(x̂GH) ṽG→H − β ρG→H.

Notes
- [·]+ denotes positive part; 1{·} is an indicator.
- Use ε ≈ 1e−3 (in traffic units) to avoid division by zero.
- If H is a true hotspot, DH > 0 by construction.