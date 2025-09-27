I would like to refactor the current Monte-Carlo Tree Search (MCTS) agent located in `src/parrhesia/flow_agent`. The existing code works, but it revealed fundamental flaws that after restudied, I would like your help to plan for perhaps a better design. I could not do it on my own. I need your help.

# About the Two-Loop Design

I believe this is a mistake. Currently in the current implementation, there are two loops: the inner loop which launches an MCTS procedure, pick a hotspot, add/remove the flows, then continue and commit to find the most optimal solution with respect to the *local objective function* (see `prompts/regulation_agent/ABOUT_OBJECTIVES.md`). As I understand it (please confirm for me about this), the backpropagation happens with respect to this *local objective function*, which is demonstrably wrong per our post-mortem analysis: the global (unscoped) objective actually increased as hotspots move from one place to another.

What is even worse, in my opinion, is that there is only one global (unscoped) objective evaluation at the end, **and it does not even get involved in the backpropagation process**. As a result, the agent **never actually learns to solve the problem beyond the local scope of greedily improve the exceedances at the hotspot it chose in the first place**.

Then there comes the problem of "budget". We currently have two level of budget control: for the inner loop and outer loop, such as maximum number of actions, wall clock time and commits (for inner loop) and regulation count (for outer loop). If we are to adopt a different approach, I believe that we have to rethink about the budget control mechanism as well (to prevent the agent from stucking in a loop).

Afterwards, the current implementation still contains some residual code from the (failed) attempt to multiprocess to speed up the search. However, as the ideas proven to be faulty, the multiprocessing code gets in the way of debugging. Thus in the new refactored approach, we will start with only one search process. All the speeding up will be left for later.

One another thing is that because there are many objective functions, it gets confusing very quickly (see `ABOUT_OBJECTIVES.md` document). This was the key culprit that failed us in the past: we were observing the wrong metric because there were many of them. In the new approach, we will use the global (unscoped) objective function as the **learning and monitoring signal**. This would reduce the confusion, and make the implementation correct.

# New Design Proposals

I would like your help in planning, in the context of the current Agent code, about the detailed approach for the following changes:

1. The general desire is to **simplify the inner loop**, so the whole implementation stays more faithful to traditional MCTS in board game playing: the agent only receives a reward at the end of the episode, where the whole plan is evaluated, and the final reward is the **global, unscoped objective function**. There will be no other "scoped" or "local" rewards. The agent will **receive no reward, until the end of the episode where the whole plan is evaluated** (equivalent to the final objective evaluation of the outer loop in the current implementation), no "mid-way" signal like the current implementation.

2. In some way, the new approach can be seen as abolishing the "inner loop", leaving just one outer loop left. We then have to rethink about budget control and reward hacking prevention: 

- We have to gate the "within-regulation" actions: when adding or removing flows, we do not allow the agent to proceed until the total number of flights in the selected flows exceeds the capacity value multiplied by some constant factor (called `entrances_to_occupancy_dwelling_factor`).

> The correct values to use should be rolling-hour occupancy count, but since this is very difficult to compute quickly, we use flight count as a quick proxy, modified by a constant factor.

- To prevent the agent from keep adding and removing flows to waste time, besides detecting loops as of the current implementation (please verify this implementation as well), we set a removal budget. Passing this removal budget, the agent can only add or continue to commit.

3. As mentioned, committing a regulation now does not lead to any evaluation (zero reward), the outer loop ticks and we move forward with a new regulation. The agent cannot modify the regulations it already committed in the past.

4. We ought to rework on the prior when it comes to adding flows (or removing flows). We will use a simpler heuristic: a more crowded the flow is (containing more flights), the more likely we are going to select it.

5. The backpropagation now happens at the leaf node (i.e., when the whole set of regulations are evaluated for global unsoped objective). We have to rethink about the fundamental state, planState and transition designs since in this new context, the outer loop works inter-regulations.

6. Don't forget about the noise at various levels: choosing hotspots, choosing flows... to encourage discovery.

7. I think we may keep the reward shaping function phi, but it does not do much in this new implementation.

# Instructions

Since this is a huge undertaking, we shall go through each task one by one. May I have your plan proposal for the most foundational part: the redesign of fundamental state, planState, transitions and other high level features after inspecting the code and reasoned over the proposals above?

# References

*For context, we refer the objective definitions per the file `ABOUT_OBJECTIVES.md`.*