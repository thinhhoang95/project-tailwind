Can you help me plan, within the contex of this code base, for an optimization framework for **entry rates** with a focus on locality remedy of the problem so the operators can amend the regulation plan iteratively.

# Definitions

- A *cell* is a pair of one traffic volume and one time slot $(s, b)$. It is a fundamental block in the 4D space-time. In the context of the project, the time bin length is controlled by `tvtw_indexer.py`, and for context (though it should be emphasized that the time length should be inferred from `tvtw_indexer` for code robustness), the time length bin is 15 minutes.

- A *hotspot* is a cell where you observe the *demand exceeding the capacity*. 

> The capacity is set hourly, and the demand is the **rolling-hour occupancy count**. These two quantities are compared with each other.

> A rolling-hour value of a quantity at time t is the cumulative sum of that quantity over the next 60 minutes. If the time bin length is 15 minutes, that is equal to summing the value over 4 time bins. 

# Motivation

The idea is to realize an optimization framework that proposes solutions to fix the problem "locally" when identified by Network Managers, rather than a "hit-and-run" approaches in academic papers. This will allow iterative problem solving, creating an opportunity to inject tacit knowledge into the solution.

Fixing a problem locally means that the optimizer will strongly focus on the parts designated by operators, while controlling for the ripple effects. Outside of these focused cells, the algorithm attempts to keep the counts as close to original as possible.

# Approach

### Attention Mechanism

The idea is to introduce the concept of attention to cells. Concretely, attention is the weight allocated to the cell when optimizing. We divide the cells into three types:

1. **Targetted Cells:** these are the cells that are in favor of heavy intervention in order to resolve a specific DCB problem. Usually reserved for hotspot, but **it is imperative that multiple cells can be selected**. Over these cells, the excess traffic is heavily penalized, while the regularization terms to keep the traffic count intact and total variation control are lower.

2. **Ripple cells:** these are the cells that are additionally added. The idea is that these cells are coupled with the targetted cells, and they get impacted as well once we decide to regulated the targetted cells. For example, if there are many flights going to the targetted hotspot from this cell, then this cell is a ripple cell of that hotspot. Like targetted cells, the excess traffic is also heavily penalized (though could be to a lesser extent than targetted cells), while the regularization terms are higher.

3. **Context cells:** Indicate all remaining cells. They usually share a very low weight for excess traffic, and a high weight for regularization terms. Context cells ensure that the "fix" stays local, otherwise the proposed regulation would "fix the problem elsewhere."

### Formal Optimization Problem for Flowless control

The optimization problem can be written formally as follows:

For each **targeted traffic volume**, we optimize for the rate $n_t$ for each time bin $t$ in the **targeted time bins**:

$$
\begin{aligned}
\min_{n}\quad & \sum_t h_t(n_t) \;+\; \beta\sum_t |n_t-n_t^{\text{old}}|\;+\;\gamma\sum_t |n_t-n_{t-1}|\\
\text{s.t.}\quad & \sum_{s\le t} n_s \le C_{hourly}(t),\quad \sum_{s\le t} n_s \le D_{hourly}(t),\quad n_t\ge 0,
\end{aligned}
$$

with a convex holding/delay proxy $h_t(\cdot)$ (e.g., piecewise-linear queue cost - practically, in meta-heuristics, we will run the delay assignment process with the FCFS queue to retrieve the concrete delay and compute the total delay imposed for this term). $C_{hourly}(t)$ is the rolling-hour capacity at time t (if the capacity changes, take the more conservative value), and the rolling-hour demand is the sum of occupancy over the next $k$ time bins (if time bin length is 15 minutes, $k=0, 1, 2, 3$).

### Flow Control

Another important concept is *flow-control*. Research has shown that rate-capping a blanket rate, blind to where the traffic came from, could likely to cause more secondary hotspots. A proper way to do this, is after having the flight list, we perform flight grouping (like community detection for instance) to obtain *flows*. In the current code base, it is handled by `flow_extractor.py` module. **Then we impose the flow rate on each flows** instead of putting a blanket cap on the whole flight list.

> In our optimization framework, we adopt a **time-varying rate** over the targetted time bins: the rate is constant across one time bin (15 minutes), but it can change every 15 minutes if needed to.

### Delay Assignment

After the rate is retrieved for each flow, an FCFS queueing delay assignment process will take place. In the current codebase, you may find it in `optimize/fcfs/scheduler.py`'s `assign_delays` function.

Then we obtain the concrete delay assignment for each flight, which is ready to be transmit to AUs.

# Instructions

1. Please reason rigorously and propose the optimization problem for "Flowful problem formulation".

1. At this moment, we are only doing the planning: please inspect the code base carefully, particularly `network_evaluator.py`, `network_plan_move.py`, `scheduler.py` and `flow_extractor.py` (`subflows`'s, NOT `flowx`) to understand the current implementations first, though significant ground work will need to be carried out.

2. 

 as well as relevant code to plan for modifications or extensions of the current code base to support our optimization framework implementation task.