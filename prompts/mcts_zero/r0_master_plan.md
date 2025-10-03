# Goal

To implement a Monte Carlo Tree Search (MCTS) for automatic regulation design as a sequential decision making process.

# Background and Goals

A *regulation plan* consists of one or many regulations. Each *regulation* is a tuple of: control traffic volume, an active time window (e.g., 9:00-10:15), an associated flight list (a flow), and the (hourly) entry rate. In Air Traffic Flow Management, when a regulation plan is submitted, the slot allocation algorithm will begin to assign delays to the flights in the "flows" (i.e., the flight lists we designated in the regulations). After delay assignment, the rolling-hour occupancy count is expected to change, some hotspots (traffic volumes + time period where exceedances happen) may extinguish, but new hotspots may emerge.

In the current project, the goal is to build a search algorithm that searches for a set of regulation plans sequentially (called a *solution*) such that exceedances in hotspots are reduced, balanced against delaying flights too much. The network begins in an initial state, then a hotspot can be selected, then flows can be extracted from that hotspot (i.e., a list of "flight clusters"), and a quick rate scan will be able to tell you how much the objective function improves. Applying a regulation plan will "step" the network into a different state (occupancy count changes, hotspot patterns may change too), and more regulations can be added sequentially, each one will result in a "network state transition", and you receive a reward (i.e., the "improvements in objective function"). 

The goal is to maximize the cumulative sum of improvements in objective function.

# Key Design Aspects

> All the implementation of the MCTS framework (hereby called **Regulation Zero**), will be in `src/regulation_zero` directory.

Note:

- One regulation = One flow.
- One regulation plan = multiple regulations = multiple flows.
- Solution = (a ordered chain of) regulation plans.
- Within one regulation plan, the order of a particular regulation is not important.
- However, the order of each plan in the solution is important and should be tracked.

### State Design

- The state `s` contains the **partial canonical solution**, and the **remaining_depth**. 

- Transition: from a network state, applying a regulation plan will result in concrete delay assignment to the flights in the flows of the regulation plan. Then the network "steps" to a new state, all hotspots need to be reidentified, and the proposal engine is called again to inspect for new proposals. Each transition will result in the objective improvement (from the corresponding selected proposal computed already by `regen`).

- Terminal condition: when remaining_depth = 0 or we ran out of hotspots, or `regen` returns nothing.

- Return: sum over all objective improvements collected along the "solution path".

- No roll-out needed. 

### Action selection

- Essentially the MCTS will "choose the hotspot-proposal". Clarification: to the MCTS, it is a "combo-action", but it consists of two smaller ones: choosing the hotspot segment (and then you have the extracted flows), followed by the proposals where you could pick any regulation plan). 

We will need to engineer the whole thing carefully, because `regen` returns `k` proposals at once, we just cache them so if we return to the same "solution state", picked the same hotspot, we don't need to call `regen` again.

- We inject some Dirichlet noise (Alpha-Zero style) when selecting the tree root (where we are about to do the combo-action) to encourage exploration.


# Instruction for Coding Agent
## About State Design
- Because we will use the `regen` (regulation generation) module, the way it works is that for each hotspot selected, it will give `k` proposals for the **regulation plan** (the whole plan, not just individual flows) (we can set `k`), along with the objective improvement. So we only need to store the proposals, the order of the plan in the solution, the associated hotspot and the rank of the proposal (`regen` ranks the `k` proposals from best to worst) to ID them, and we can use the ID as part of the canonical solution.
- The `remaining_depth` is just the maximum number of regulation plans a solution allow subtracting the `order` of the regulation plan.
- I would like you to reason how we should design this concretely, including the data structure (if necessary), with regard to the transposition table.
A succinct example (1 transition only) involving selecting hotspots, call `regen` proposals, transition, compute objective function can be seen in `regen_second_order.py`.
