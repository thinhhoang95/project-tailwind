# Feature Engineering for Fast Heuristics for Hot‑spot Regulation (Meta-Optimization): A Structured Approach Using Matched‑filter Scoring, Union‑/Separate‑Cap Classification

## 1 High‑level Intuition  

- The relevant correlation is between a flow’s negative‑lobe time series at the hotspot’s reference time (shifted by the flow’s offsets) and the hotspot’s exceedance profile. The negative lobe earns a safety reward; the positive lobe risks secondary hotspots. Both are flow‑wise rigid translates under ground delay. Hence the fast decision reduces to a **matched‑filter problem**: select flows whose removal profile is phase‑aligned with the hotspot overload while avoiding re‑injection into tight downstream slack.  

- Constant union caps versus separate constant caps have no universal dominance. Pooling provides adaptivity in time; separation yields additivity across multi‑sector prices. This dichotomy yields a clean phase diagram for when to group flows under a union cap and when to keep them separate.  

- These observations lead to a natural family of surrogate features:  

  1. **Flow‑hotspot affinity** via a cross‑correlation‑like “price kernel’’ $v_{G}(t)$ evaluated at the right phase $t = t^{\star} - \tau_{G,s^{\star}}$;  
  2. **Re‑injection risk** via the reachable slack profile $\operatorname{Slack}_{G}(t)$;  
  3. **Pooling vs. separation indices** between flows (temporal overlap, offset orthogonality, slack misalignment, price heterogeneity).  

  All are cheap to compute, interpretable, and can be fed directly into an online‑learning policy.  

## 2 Plan / Strategy  

We explore three families of fast heuristics (all pre‑KKT, compatible with the Tier‑2 calculus).  

### 2.1 Matched‑filter Scoring  

For each hotspot cell $H = (s^{\star}, t^{\star})$ we compute a **unit‑marginal benefit**

$$
v_{G}(t)=\sum_{s}\bigl[w_{\text{sum}}+w_{\max}\,\theta_{s,\,t+\tau_{G,s}}\bigr]\,
\mathbf{1}\bigl\{o_{s}(t+\tau_{G,s})>0\bigr\}
$$

and a **reachable downstream slack**

$$
\operatorname{Slack}_{G}(t)=\min_{s}s_{s}\bigl(t+\tau_{G,s}\bigr).
$$

1. **Phase alignment.** Any unit of removal applied at reference time $t$ maps to a unit reduction at the hotspot time $t+\tau_{G,s^{\star}}$. Hence the relevant phase is  

$$
t_{G}:=t^{\star}-\tau_{G,s^{\star}}
$$

and the unit marginal benefit is $v_{G}(t_{G})$.

> Notice: the delay that we consider in this approach is **ground-delay**, thus if a flight is delayed, all the traffic volumes it passes through will be affected. But the flow will be throttled (through its rate) at a specific traffic volume (called a *control volume*). As a result, $\tau_{G, s}$ could either be positive or negative.

2. **Cell‑specific unit price.** Define  

$$
\begin{aligned}
v_{G\rightarrow H}:=&\;
\bigl[w_{\text{sum}}+w_{\max}\,\theta_{s^{\star},t^{\star}}\bigr]\,
\mathbf{1}\bigl\{o_{s^{\star}}(t^{\star})>0\bigr\}\\
&+\kappa\!\sum_{s\neq s^{\star}}\!
\bigl[w_{\text{sum}}+w_{\max}\,\theta_{s,\,t^{\star}+\tau_{G,s}-\tau_{G,s^{\star}}}\bigr]\,
\mathbf{1}\bigl\{o_{s}(t^{\star}+\tau_{G,s}-\tau_{G,s^{\star}})>0\bigr\},
\end{aligned}
$$

where $\kappa\in[0,1]$ controls how much collateral assistance we credit. Setting $\kappa=0$ yields a purely local score.  

3. **Eligibility indicator.** Define an activity index  

$$
a_{G\rightarrow H}:=
\mathbf{1}\!\bigl\{x_{G}(t_{G})\ge q_0 \bigr\},
$$

where $q_0$ is a parameter that controls the minimum amount of traffic required in order for the flow "to be taken seriously." We have to do this because otherwise, the flow could have hardly any traffic at all, but the unit cost is so good leading to its selection.

$$
a_{G\rightarrow H}:=
\sigma\!\bigl(\gamma\bigl(x_{G}(t_{G})-q_0 \bigr)\bigr),
$$

with $\sigma(\cdot)$ the logistic sigmoid.  

4. **Slack‑risk penalty.**  

$$
\rho_{G\rightarrow H}:=
\Bigl[1-\frac{\operatorname{Slack}_{G}(t_{G})}{S_{0}}\Bigr]_{+},
$$

where $S_{0}$ is a normalising slack unit which is also 

5. **Matched‑filter net score.**  

$$
\operatorname{Score}(G\mid H)=\alpha\,a_{G\rightarrow H}\,v_{G\rightarrow H}
\;-\;\beta\,\rho_{G\rightarrow H},
$$

with tunable weights $\alpha,\beta\ge0$ (learnable). An optional delay cost can be subtracted as $-\lambda_{\text{delay}}$ per minute of removal.  

### 2.2 Union‑cap vs. Separate‑caps Classifier (Constant Caps)  

Given a short list of top flows $\{G_{i}\}$ for hotspot $H$, we decide whether to pool them under a single constant **union cap** or keep **separate constant caps**.  

#### Diagnostic features (pairwise)

*Temporal overlap*  

$$
\operatorname{Overlap}_{ij}:=\sum_{t\in\mathcal{W}}\min\{x_{i}(t),\,x_{j}(t)\}.
$$

*Offset orthogonality*  

$$
\operatorname{Orth}_{ij}:=
1-\frac{\bigl|\{s: o_{s}(t^{\star}+\tau_{G_{i},s}-\tau_{G_{i},s^{\star}})>0\}
\cap
\{s: o_{s}(t^{\star}+\tau_{G_{j},s}-\tau_{G_{j},s^{\star}})>0\}\bigr|}
{\bigl|\{s: o_{s}(\cdot)>0\}\bigr|}.
$$

*Slack misalignment* (Pearson correlation on a window around the phases)  

$$
\operatorname{SlackCorr}_{ij}:=
\operatorname{corr}\bigl(\operatorname{Slack}_{G_{i}}(\cdot),\,
\operatorname{Slack}_{G_{j}}(\cdot)\bigr).
$$

*Price heterogeneity*  

$$
\operatorname{PriceGap}_{ij}:=
\frac{\bigl|v_{G_{i}}(t_{G_{i}})-v_{G_{j}}(t_{G_{j}})\bigr|}
{v_{G_{i}}(t_{G_{i}})+v_{G_{j}}(t_{G_{j}})+\varepsilon}.
$$

#### Decision rule  

With learnable thresholds $(\tau_{\mathrm{ov}},\tau_{\mathrm{sl}},\tau_{\mathrm{pr}},\tau_{\text{orth}})$:

1. **Union preferred** if  
   $$
   \operatorname{Overlap}_{ij}\le\tau_{\mathrm{ov}},\quad
   \operatorname{SlackCorr}_{ij}\ge\tau_{\mathrm{sl}},\quad
   \operatorname{PriceGap}_{ij}\le\tau_{\mathrm{pr}}.
   $$

2. **Separate caps favored** if  
   $$
   \operatorname{Orth}_{ij}\ge\tau_{\text{orth}}
   \quad\text{or}\operatorname{PriceGap}_{ij}\ge\tau_{\mathrm{pr}}.
   $$

3. **Tie‑break** by comparing a quick estimate of waste versus price gain for the two alternatives (e.g., using the pre‑computed $v_{G}$ values).

For three or more flows we apply the pairwise rule agglomeratively (simple clustering on a distance $d$ derived from the diagnostics).  
