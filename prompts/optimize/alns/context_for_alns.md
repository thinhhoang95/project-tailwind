Thanks! I’ll review your codebase—starting from `tests/test_regulation_move_integration.py`—to understand how `NetworkPlan` works, how to evaluate the objective function, and how to implement a base version of ALNS using the `alns` library with a couple of destroy/repair moves.

I’ll get back to you shortly with a detailed implementation plan and a minimal working example.


# Adaptive Large Neighborhood Search (ALNS) for Network Plan Optimization

## Step-by-Step Plan

1. **Solution Representation (Network Plan):** The solution in this problem is a **NetworkPlan** – a collection of traffic flow regulations. In code, `NetworkPlan` is a simple container for multiple `Regulation` objects (or their defining strings). It supports operations to add or remove regulations dynamically, which will be crucial for defining our ALNS moves. Each regulation specifies a traffic volume (sector) and a rate limit over certain time windows (with optional filters for flight origin/destination). For example, a regulation string like `"TV_MASH5RL IC__ 10 36-45"` represents a regulation at traffic volume **MASH5RL** with no special filter (`IC__` denotes a wildcard filter), a rate of 10 flights per time window, active in time windows 36–45. The **NetworkPlan** will hold such regulations, and its state can be modified by our heuristic operators.

2. **Objective Function Definition:** We define the objective as a single float value that we aim to **minimize**. In this ATFM context, the objective combines measures of **excess traffic** (traffic beyond capacity) and the **cost of regulations** (delays imposed). The code already provides an `OptimizationProblem.objective()` method that computes this score. It works by evaluating the network state (flights and regulations applied) through a `NetworkEvaluator` to get metrics like `z_95` (a high-percentile or max overload metric) and `z_sum` (total overload), plus the total delay in seconds. The objective is a weighted sum: for example, using weights from the test (`objective_weights = {'z_95':1.0, 'z_sum':1.0, 'delay':0.01}`), the objective might be calculated as:

   $$
   \text{Objective} = 1.0 \times z_{95} \;+\; 1.0 \times z_{sum} \;+\; 0.01 \times \frac{\text{total\_delay\_seconds}}{60}
   $$

   In code this is exactly how the value is returned. A lower objective means less overload and less delay cost, so ALNS will try to find a network plan that minimizes this score.

3. **Destroy and Repair Operators:** We will implement a basic set of ALNS neighborhood operators that **modify the NetworkPlan**. In a large neighborhood search, a **destroy operator** partially destroys the current solution (e.g. removes some regulations), and a **repair operator** then fixes or extends it (e.g. adds new regulations). For the base version, we can start with two simple moves:

   * **Destroy Operator – Remove a Regulation:** This operator will remove one regulation from the current network plan at random. If the plan currently has regulations, we pick a random index and drop that regulation. (If the plan is empty, this operator can just return the state unchanged, as there’s nothing to remove.) The `NetworkPlan.remove_regulation(idx)` method makes this removal straightforward. By removing a regulation, we potentially allow more traffic (increasing overload) but also eliminate its delay cost – the ALNS will explore if this leads to a better trade-off in the objective.
   * **Repair Operator – Add a Regulation:** This operator will add a new randomly-generated regulation to the network plan. We need to create a valid `Regulation` string with a random traffic volume and rate. To ensure validity, we should pick a **traffic volume ID** that exists in our data and a reasonable **rate** value. For example, we can sample a traffic volume from the list of IDs known to the system (the `TVTWIndexer` contains a mapping of volume IDs) and choose a random rate (e.g. between 5 and 60 flights per hour, as seen in test scenarios). We’ll also decide on random time windows for the regulation (e.g. a random start within the day and a random duration). Using these, we form a regulation string like `"TV_<VolumeID> IC__ <Rate> <StartTW>-<EndTW>"` – here `IC__` indicates no origin/dest filter (wildcard) and `<StartTW>-<EndTW>` is the active time window range. We then call `NetworkPlan.add_regulation(new_reg_str)` to add this regulation to the plan. This introduces a new capacity restriction (which can reduce overload at the cost of added delay).

   *Implementation note:* Both operators will create a **new candidate state** (a new NetworkPlan or a copy) rather than modifying the original in-place, because the ALNS library expects the operators to return a new `State`. We may represent the state as a custom class (e.g. `NetworkPlanState`) holding the `NetworkPlan` and all necessary context (flight list, evaluator, etc.) to compute the objective. Its `objective()` method will apply the network plan to a copy of the baseline flight data and compute the resulting objective score. For instance, it can deep-copy the baseline `FlightList`, apply all regulations via the `NetworkPlanMove` (which runs the delay simulation for each regulation), and then use `NetworkEvaluator` to compute new `z_95`, `z_sum`, and delays for that state.

4. **Initializing the ALNS Framework:** With the solution representation and operators ready, we set up the ALNS solver using the `alns` library. We start by creating an `ALNS` instance (e.g. `alns = ALNS()`). Next, we **register our operators** with this ALNS instance:

   * Use `alns.add_destroy_operator(remove_random_reg, name="RemoveReg")` to add our removal heuristic.
   * Use `alns.add_repair_operator(add_random_reg, name="AddReg")` to add the insertion heuristic.

   Here, `remove_random_reg` and `add_random_reg` would be the Python functions we defined for the destroy and repair moves. The ALNS library will treat these as the pool of operators to use. We should also construct the **initial solution state** at this point. A natural starting point is the **baseline scenario with no regulations**, since initially no traffic flow restrictions are applied. We can create an empty NetworkPlan (e.g. `initial_plan = NetworkPlan([])` which results in a plan of length 0). Then we wrap it in our state class (so that we have access to the flight data and an objective method). We can compute and note the baseline objective value at this stage for reference – this is the score with zero regulations.

5. **Operator Selection and Acceptance Criteria:** ALNS will iteratively select a destroy and repair pair to apply at each iteration. We should choose a strategy for how operators are picked (**operator selection scheme**) and which new solutions to accept (**acceptance criterion**):

   * For **operator selection**, a common choice is the **roulette wheel selection** that adapts operator weights based on past performance. Initially, each operator (RemoveReg, AddReg, etc.) can have equal weight. After each iteration, the weight is updated depending on whether the move led to a new global best, an improvement, an accepted (but not improved) solution, or a rejected solution. For example, we might assign scores like 5 for a global best, 2 for an improvement, 1 for an accepted, and 0.5 for a rejected move, and use a decay factor to gradually forget old performance. The `RouletteWheel` selection scheme in the `alns.select` module can be configured with these parameters (`scores`, `decay`, and the number of destroy/repair operators).
   * For the **acceptance criterion**, in a basic ALNS we can start with a greedy **Hill Climbing** rule (only accept the new solution if it has an equal or lower objective than the current) to focus on improvements. This is implemented by the `HillClimbing` class in `alns.accept`, which “only accepts better solutions”. Hill-climbing is simple but can get stuck in local optima, so you might alternatively use a **Simulated Annealing** criterion that occasionally accepts worse solutions to escape local minima. The `alns.accept` module provides a `SimulatedAnnealing` class we can configure with an initial temperature and cooling rate if more exploration is needed. For the initial implementation, HillClimbing (greedy acceptance) is straightforward.

6. **Running the ALNS Search:** Finally, we execute the ALNS algorithm with our setup. We call `result = alns.iterate(initial_state, op_select, accept, stop)` where:

   * `initial_state` is our starting solution (the state wrapping the empty NetworkPlan and baseline flights).
   * `op_select` is the operator selection scheme (e.g. the configured RouletteWheel).
   * `accept` is the acceptance criterion (e.g. HillClimbing).
   * `stop` is a stopping criterion. We can use a simple **iteration limit** (for example, stop after a fixed number of iterations or after a certain time). The library provides `MaxIterations(n)` for this purpose. For a minimal example, even 100 or 500 iterations might be enough to test the setup.

   The `.iterate` method will loop, at each iteration randomly selecting our destroy and repair pair, applying them to the current solution to get a new candidate network plan, evaluating the objective, and deciding whether to accept it. Over iterations, the adaptive mechanism will favor operators that lead to improvements. The result is returned as a `Result` object containing the best solution found.

7. **Analyzing Results:** From the `Result`, we can retrieve the best solution state via `best_state = result.best_state`. We can inspect its objective value (`best_state.objective()`) and the set of regulations it contains. For instance, we might print the best objective and list the regulations (as strings) that constitute the optimal NetworkPlan. In the testing code, after applying moves they recompute the objective and improvement – similarly, we can compare the best found objective to the baseline to see the gain. We should also verify that all regulations in the final plan are valid (they should be, since our moves always generate valid ones by construction).

By following these steps, we establish an ALNS framework: starting from no regulations, the algorithm will try removing and adding regulations (in this case, mostly adding since we start empty) to find a set that reduces network overload at minimal delay cost. The adaptive selection will inform which move (adding or removing) is more effective as the search progresses.

## Minimal ALNS Example Usage

Below is a simplified example (in code form) demonstrating how one might set up the ALNS loop with the `alns` library for this problem. This assumes that the necessary data (traffic volumes, flights, indexer, etc.) have been loaded and that we have the `NetworkPlanMove` logic available to apply regulations and compute the objective.

```python
from alns import ALNS
from alns.accept import HillClimbing  # or SimulatedAnnealing for a non-greedy approach
from alns.select import RouletteWheel
from alns.stop import MaxIterations

# Suppose we have these already prepared from the setup (similar to the test code):
traffic_volumes_gdf = ...        # loaded GeoDataFrame of traffic volumes (with capacities)
indexer = ...                    # TVTWIndexer loaded with traffic volume IDs
flight_list = ...                # FlightList loaded with flights and occupancy data
parser = RegulationParser(flights_file, indexer)
objective_weights = {'z_95': 1.0, 'z_sum': 1.0, 'delay': 0.01}
horizon = 100
# We'll create a state class to hold solution and compute objective:
class NetworkPlanState:
    def __init__(self, network_plan, base_flight_list):
        self.network_plan = network_plan              # current set of regulations
        self.base_flight_list = base_flight_list      # reference baseline flight data
    def objective(self) -> float:
        # Create a copy of the flight list to apply current regulations
        current_flights = deepcopy(self.base_flight_list)
        # Apply all regulations in the network plan to the flights (calculate delays)
        move = NetworkPlanMove(self.network_plan, parser=parser,
                                flight_list=current_flights, tvtw_indexer=indexer)
        move(current_flights)  # apply in-place
        # Compute metrics on the updated flight list
        evaluator = NetworkEvaluator(traffic_volumes_gdf, current_flights)
        metrics = evaluator.compute_horizon_metrics(horizon)
        # Weighted sum objective (using same weights as optimization_problem)
        return (objective_weights['z_95'] * metrics["z_95"] 
                + objective_weights['z_sum'] * metrics["z_sum"] 
                + objective_weights['delay'] * (metrics["total_delay_seconds"] / 60.0))

# Initial solution: start with no regulations
initial_plan = NetworkPlan([])  # empty plan
initial_state = NetworkPlanState(initial_plan, flight_list)
print("Baseline objective (no regulations):", initial_state.objective())

# Define destroy and repair operators:
def remove_random_reg(state: NetworkPlanState, rng):
    # Remove a random regulation (if any exist)
    if len(state.network_plan) == 0:
        return state  # nothing to remove
    new_plan = NetworkPlan([reg.raw_str for reg in state.network_plan.regulations])  # copy current regs
    idx = rng.integers(0, len(new_plan))  # random index to remove
    new_plan.remove_regulation(idx)       # perform removal
    return NetworkPlanState(new_plan, state.base_flight_list)

def add_random_reg(state: NetworkPlanState, rng):
    # Add a new random regulation
    new_plan = NetworkPlan([reg.raw_str for reg in state.network_plan.regulations])  # copy current regs
    # Randomly choose a traffic volume and rate
    tv_id = rng.choice(list(indexer._tv_id_to_idx.keys()))
    rate = int(rng.integers(5, 60))      # random capacity rate between 5 and 60
    # Choose a random time window interval within the horizon
    start_tw = int(rng.integers(0, horizon - 5))
    end_tw = start_tw + int(rng.integers(1, 6))  # random duration up to 5 windows
    # Form the regulation string (no filter condition)
    reg_str = f"TV_{tv_id} IC__ {rate} {start_tw}-{end_tw}"
    new_plan.add_regulation(reg_str)     # add the new regulation
    return NetworkPlanState(new_plan, state.base_flight_list)

# Set up ALNS solver
alns = ALNS()
alns.add_destroy_operator(remove_random_reg)
alns.add_repair_operator(add_random_reg)

# Use a roulette wheel selection scheme and a simple Hill Climbing acceptance
op_select = RouletteWheel(scores=[5, 2, 1, 0.5], decay=0.8, num_destroy=1, num_repair=1)
accept = HillClimbing()  # only accept improvements
stop = MaxIterations(500)  # stop after 500 iterations

# Run the ALNS iteration
result = alns.iterate(initial_state, op_select, accept, stop)
best_state = result.best_state  # best solution found

# Output the results
print("Best objective found:", best_state.objective())
best_plan = best_state.network_plan
print(f"Best NetworkPlan has {len(best_plan)} regulations:")
for reg in best_plan:
    print(" -", reg.raw_str)
```

In this example, we created a `NetworkPlanState` class with an `objective()` method so that it conforms to the `alns.State` interface. The destroy operator (`remove_random_reg`) and repair operator (`add_random_reg`) both produce a new `NetworkPlanState` by copying the current plan and removing or adding a regulation, respectively. We used the `TVTWIndexer` (`indexer`) to get a valid traffic volume ID for new regulations and ensured the rate is a positive integer. The time window selection is random but within the defined horizon.

After adding our operators to the `ALNS` instance, we selected a basic **RouletteWheel** strategy for operator selection (with some example scores and decay factor) and a **HillClimbing** acceptance criterion to keep only improving solutions. We then ran 500 iterations of ALNS. Finally, we retrieved the best solution found and printed its objective value and the list of regulations in that network plan.

This minimal setup should yield a working ALNS loop that starts from an empty plan and iteratively tries adding or removing regulations to minimize the objective score. As the integration test indicated, applying a set of regulations and recomputing the objective will show a difference (improvement) from the baseline. With the ALNS search, we automate this process to find a near-optimal combination of regulations.
