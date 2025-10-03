# Documentation for Flow Basket Features

This document provides a detailed explanation of the setwise features computed in the `flow_basket_features.py` script. The script serves as an end-to-end example of using the Parrhesia MetaOpt library to analyze and compare traffic flows around a congested hotspot.

## Core Concepts

Before diving into the features, it's helpful to understand some core concepts used in the calculations.

-   **Flow**: A group of flights that share similar trajectory characteristics. In the script, we compare two flows: an "anchor" flow and the "largest" flow (treated as a set of flights).
-   **Hotspot**: A specific airspace volume (Traffic Volume or TV) at a specific time bin that is experiencing congestion.
-   **Control Volume**: A TV that is considered representative of a flow's route. A flow's demand is typically measured at its control volume.
-   **Demand Series (`xG`)**: A time series representing the number of flights in a flow `G` that are scheduled to be at their control volume at each time bin.
-   **Travel Time Map (`τ`, `tau_row_to_bins`)**: For a given flow, this map stores the travel time (in discrete time bins) from the flow's control volume to any other TV `s`. `τ_{G,s}` represents this travel time.
-   **Phase Time (`t_G`)**: This is a crucial concept for aligning flows. It represents the time at which a flow `G` is at its control volume, such that it arrives at the hotspot volume `s*` at the hotspot time `t*`. It's calculated as: `t_G = t* - τ_{G,s*}`. This alignment allows for a meaningful comparison of flows relative to a common congestion event.
-   **Slack**: A measure of available capacity in a TV at a given time. Low or negative slack indicates high congestion. This is stored in the `slack_per_bin_matrix`.
-   **Hourly Excess (`hourly_excess_bool`)**: A boolean matrix indicating whether a TV is overloaded (demand exceeds capacity) in a given hour-long period.

## Setwise Features

The script computes four "setwise" features that compare the anchor flow to the largest flow (the "set"). These features provide a multi-faceted view of how similar or different the two flows are in terms of their temporal, spatial, and congestion impact.

---

### 1. Temporal Overlap (`overlap`)

#### Informal Explanation

This feature measures the degree of temporal competition between two flows. It quantifies how much their demands at their respective control volumes overlap in time, after being aligned to the hotspot's timeframe. A high value signifies that both flows are demanding resources concurrently, potentially leading to interference.

#### Detailed Calculation

The calculation is performed by the `temporal_overlap` function in `pairwise_features.py`.

1.  **Phase Alignment**: The demand series of the "set" flow (`x_set`) is first shifted in time to align its phase with the "anchor" flow. The time shift `d` is the difference in their phase times: `d = tG_anchor - tG_set`. This results in a shifted demand series, `x_set_shift`.
2.  **Overlap Computation**: The overlap is then calculated as the sum of the minimum of the two demand series at each time bin, within a specified window `W` around the anchor's phase time:
    \[ \text{Overlap} = \sum_{t \in W} \min(x_{\text{anchor}}(t), x_{\text{set\_shift}}(t)) \]

#### Edge Cases

-   If the time window `W` is empty or does not contain any valid time bins for the demand series, the overlap is 0.
-   If the demand series have different lengths, the comparison is performed over the length of the shorter series.


#### An Example
Imagine the following scenario:
A hotspot occurs at TV H at time bin t* = 100.
Anchor Flow:
Control Volume: TV A
Travel time from A to H: 5 bins
Phase Time tG_anchor = 100 - 5 = 95
Its demand x_anchor peaks at time 95.
Set Flow:
Control Volume: TV B
Travel time from B to H: 8 bins
Phase Time tG_set = 100 - 8 = 92
Its demand x_set peaks at time 92.
Without alignment, their demand peaks are at different times (92 and 95). But both are timed to arrive at the hotspot at the same time (100).
Alignment Calculation:
Time shift d = tG_anchor - tG_set = 95 - 92 = 3 bins.
We shift the x_set demand series forward by 3 bins.
The peak of the new x_set_shift is now at time 92 + 3 = 95.
Now, both the anchor flow's demand and the set flow's aligned demand peak at the same time, t=95. When we calculate the min() of the two series, we will correctly capture their temporal competition relative to the hotspot.

---

### 2. Offset Orthogonality (`orth`)

#### Informal Explanation

This feature assesses the spatial separation of two flows by looking at the congested airspace they are predicted to traverse. A value of 1 indicates perfect orthogonality (the flows do not pass through any of the same future congested areas), while a value of 0 means they pass through the exact same set of congested areas, suggesting a high potential for spatial conflict.

#### Detailed Calculation

This is calculated by the `offset_orthogonality` function in `pairwise_features.py`.

1.  **Identify Overloaded TVs**: For each flow (`i` and `j`), the algorithm identifies the set of TVs (`T_i` and `T_j`) that are in an overloaded state (`hourly_excess_bool` is true) at the precise time the flow is expected to pass through them. This time is calculated relative to the hotspot time (`t*`) and the flow's travel time map (`τ`): `t_{s} = t* + τ_{i,s} - τ_{i,s*}` for each TV `s`. The hotspot TV itself is excluded from these sets.
2.  **Jaccard Distance**: The orthogonality is the Jaccard distance between these two sets of overloaded TVs:
    \[ \text{Orth} = 1 - \frac{|T_i \cap T_j|}{|T_i \cup T_j|} \]
    where `∩` is the intersection and `∪` is the union of the two sets.

#### Edge Cases

-   If neither flow passes through any overloaded TVs (`|T_i ∪ T_j| = 0`), they are considered perfectly orthogonal, and the function returns 1.0.

---

### 3. Slack Correlation (`slack_corr`)

#### Informal Explanation

This feature compares the congestion patterns experienced by two flows over time. It does this by correlating their "slack profiles." A slack profile represents the minimum available capacity (slack) a flow encounters across all the airspace it occupies, tracked over a time window.
-   A **high positive correlation (~1)** suggests that both flows tend to experience congestion (low slack) and free capacity (high slack) at the same times.
-   A **high negative correlation (~-1)** suggests their congestion experiences are inverse; when one hits traffic, the other has clear skies.
-   A **correlation near 0** means there is no linear relationship between their experienced congestion patterns.

#### Detailed Calculation

The `slack_corr` function in `pairwise_features.py` performs this calculation.

1.  **Generate Slack Profiles**: For each flow, a slack profile is generated using the `slack_profile` function. This function computes the minimum slack encountered by the flow across all TVs it touches, for each time step `Δ` in a window around the flow's phase time `t_G`. The slack for a specific TV `s` at a specific time is looked up from `slack_per_bin_matrix` at the aligned time `t_G + Δ + τ_{G,s}`. This process yields two time series of minimum slack values, `Si` and `Sj`.
2.  **Pearson Correlation**: The Pearson correlation coefficient is then computed between the two slack profiles `Si` and `Sj`.

#### Edge Cases

-   If either slack profile is empty or they have different lengths, the correlation is 0.
-   If either profile has zero variance (i.e., the slack values are constant), the correlation is 0 to prevent division by zero in the Pearson formula.

---

### 4. Price Gap (`price_gap`)

#### Informal Explanation

This feature measures the relative difference in the "congestion price" of two flows. This "price," denoted as `v_tilde` (`ṽ`), quantifies a flow's contribution to congestion at and around the hotspot. It's a sophisticated metric that considers not just the flow's own demand but also the state of the system it flies through. A low price gap indicates that both flows have a similar impact on system congestion.

#### Detailed Calculation

The calculation is handled by `price_gap` in `pairwise_features.py`, which relies on `price_contrib_v_tilde` from `per_flow_features.py`.

1.  **Compute Congestion Price (`ṽ`)**: The `price_contrib_v_tilde` function calculates the congestion price for each flow. The calculation is intricate, but can be summarized as follows:
    -   It computes an "addressable share" (`ω`) for each TV the flow touches. This share is `min(1, x_sG / (D_s + ε))`, where `x_sG` is the flow's demand at the TV and `D_s` is the capacity exceedance (how much demand is over capacity). This measures how much of the congestion a flow is responsible for.
    -   The final price `ṽ` is a weighted sum of unit prices for congestion. It has two components:
        -   **Primary**: The price at the hotspot TV, weighted by the flow's addressable share there.
        -   **Secondary (Ripple)**: A discounted (`κ`) sum of prices at other congested TVs the flow touches, also weighted by their addressable shares.
2.  **Calculate the Gap**: Once `ṽ_anchor` and `ṽ_set` are computed, the price gap is calculated as a normalized absolute difference:
    \[ \text{PriceGap} = \frac{| \tilde{v}_{\text{anchor}} - \tilde{v}_{\text{set}} |}{ \tilde{v}_{\text{anchor}} + \tilde{v}_{\text{set}} + \epsilon } \]
    where `ε` is a small constant to ensure numerical stability.

#### Edge Cases

-   If the sum of the two prices is near zero (i.e., neither flow contributes to congestion), the price gap is returned as 0.
