# MCTS Agent for Flow Regulation: Detailed Documentation

This document provides a detailed explanation of the Monte Carlo Tree Search (MCTS) agent designed for discovering and implementing air traffic flow management regulations.

## 1. Overview

The MCTS agent is an autonomous system that identifies traffic hotspots and formulates regulations to mitigate congestion. It uses MCTS to explore a vast search space of possible regulations and selects those that provide the most significant improvement to network conditions. The agent is designed to be data-driven, using flight data, traffic volume capacities, and network evaluators to make informed decisions.

The process can be broken down into these main steps:
1.  **Hotspot Discovery**: The agent first analyzes traffic to identify areas and times of high congestion (hotspots).
2.  **MCTS Search**: For a chosen hotspot, the agent runs MCTS to search for an effective regulation. This involves selecting flows of traffic to control and determining the optimal rate for these flows.
3.  **Regulation Commit**: The best regulation found during the search is committed and added to a plan.
4.  **Iteration**: The agent can repeat the process to create multiple regulations, building a comprehensive traffic management plan.
5.  **Final Evaluation**: Once the plan is complete, a final objective score is computed to measure its overall effectiveness.

## 2. Core Components

The agent is composed of several interacting components.

### `MCTSAgent`

This is the high-level orchestrator. It manages the overall process from hotspot discovery to final plan evaluation.

-   **Inputs**:
    -   `evaluator: NetworkEvaluator`: Simulates network conditions.
    -   `flight_list: FlightList`: Provides access to flight and occupancy data.
    -   `indexer: TVTWIndexer`: Maps traffic volumes and time windows to indices.
    -   Configuration objects (`MCTSConfig`, `RateFinderConfig`, `HotspotDiscoveryConfig`).
    -   `logger: SearchLogger` (optional): For logging search progress.
    -   `max_regulations: int` (optional): The maximum number of regulations to create.
-   **Output**: A tuple of `(PlanState, RunInfo)`.
    -   `PlanState`: The final state of the plan, containing the list of committed regulations.
    -   `RunInfo`: A dataclass with summary information about the run, including the number of commits, total objective improvement, log file path, and a summary dictionary.

### `MCTS`

This class implements the core MCTS algorithm with progressive widening and potential shaping. It explores the action space to find the best `CommitRegulation` action.

-   **Key features**:
    -   **Progressive Widening**: Controls the branching factor of the search tree, preventing it from becoming too wide too early.
    -   **PUCT (Polynomial Upper Confidence Trees)**: The selection strategy used to balance exploration and exploitation.
    -   **Potential Shaping**: A reward shaping technique (`_phi` function) that provides intermediate rewards to guide the search towards promising states (states with lower traffic overload).
    -   **Commit Evaluation Caching**: Caches the expensive results from `RateFinder` to speed up simulations.

### `PlanState`

This dataclass represents the state of the search. It is immutable, and new states are created by applying actions.

-   **Key attributes**:
    -   `plan: List[RegulationSpec]`: The sequence of regulations committed so far.
    -   `stage: str`: The current stage of building a regulation (e.g., `idle`, `select_hotspot`, `select_flows`, `confirm`).
    -   `hotspot_context: HotspotContext`: Information about the current hotspot being considered.
    -   `metadata: dict`: Stores auxiliary information, such as the list of candidate hotspots.
    -   `z_hat: np.ndarray`: A proxy for network overload, used by the potential shaping function.

### Actions

Actions define the possible moves within the search tree. They are simple dataclasses.

-   `NewRegulation`: Start defining a new regulation.
-   `PickHotspot`: Select a hotspot (a traffic volume and time window) to regulate.
-   `AddFlow`: Add a flow of traffic to the regulation being built.
-   `RemoveFlow`: Remove a flow from the regulation.
-   `Continue`: Proceed to the confirmation stage with the selected flows.
-   `CommitRegulation`: Finalize the regulation and evaluate its optimal rates. This is a terminal action in a sub-search, and triggers `RateFinder`.
-   `Back`: Go back to the previous stage.
-   `Stop`: Abort the current regulation and return to the `idle` state.

### `RateFinder`

This component is responsible for finding the optimal control rates for a given regulation (a hotspot and a set of flows). It is called when a `CommitRegulation` action is evaluated.

-   It searches for rates that maximize the objective improvement (`delta_j`).
-   It can perform either a per-flow rate optimization or a blanket rate for all selected flows.

### `HotspotInventory` and `HotspotDiscovery`

Before the MCTS search begins, the `MCTSAgent` uses `HotspotInventory` to find candidate hotspots. The discovery process is configured by `HotspotDiscoveryConfig`.

-   It identifies traffic volumes and time windows with significant overload.
-   For each hotspot, it identifies the major contributing flows of traffic using graph clustering algorithms (e.g., Leiden).
-   The output is a list of candidate hotspots that are provided to the MCTS search via the initial `PlanState.metadata`.

## 3. Configuration

The agent's behavior is controlled by three main configuration dataclasses.

### `MCTSConfig`

Parameters for the MCTS search itself.

-   `c_puct: float`: Exploration constant in the PUCT formula. Higher values encourage more exploration.
-   `alpha: float`, `k0: int`, `k1: float`: Parameters for progressive widening.
-   `commit_depth: int`: The number of `CommitRegulation` actions the agent will take in a single `MCTSAgent.run()` call.
-   `max_sims: int`: The number of MCTS simulations to run for each regulation search.
-   `max_time_s: float`: Maximum time in seconds for a search.
-   `commit_eval_limit: int`: A budget for how many times the expensive `RateFinder` can be called per search.
-   `priors_temperature: float`: Controls the sharpness of the softmax for priors.
-   `phi_scale: float`: Scaling factor for the potential shaping reward.
-   `seed: int`: RNG seed for reproducibility.

**Example:**
```python
mcts_cfg = MCTSConfig(
    max_sims=12,
    commit_depth=1,
    commit_eval_limit=6,
    seed=0
)
```

### `RateFinderConfig`

Parameters for the `RateFinder`.

-   `use_adaptive_grid: bool`: Whether to use an adaptive grid search for finding rates.
-   `max_eval_calls: int`: The budget of evaluator calls for the rate search.

**Example:**
```python
rf_cfg = RateFinderConfig(use_adaptive_grid=True, max_eval_calls=128)
```

### `HotspotDiscoveryConfig`

Parameters for the initial hotspot discovery phase.

-   `threshold: float`: Minimum overload threshold for a segment to be considered part of a hotspot.
-   `top_hotspots: int`: The maximum number of hotspots to identify.
-   `top_flows: int`: The maximum number of candidate flows to identify for each hotspot.
-   `max_flights_per_flow: int`: The maximum number of flights to consider within each flow.
-   `leiden_params: dict`: Parameters for the Leiden community detection algorithm used for flow clustering.
-   `direction_opts: dict`: Options for considering flight directions.

**Example:**
```python
disc_cfg = HotspotDiscoveryConfig(
    threshold=0.0,
    top_hotspots=5,
    top_flows=3,
    max_flights_per_flow=15,
    leiden_params={"threshold": 0.3, "resolution": 1.0, "seed": 0},
    direction_opts={"mode": "none"},
)
```

## 4. The MCTS Algorithm in Detail

### The Simulation Loop

Each simulation traverses the tree from the root to a leaf node, then backs up the result.

1.  **Selection**: Starting from the root, the algorithm recursively selects the child with the highest PUCT score until an unexpanded node is reached. The PUCT score for an action `a` from state `s` is:
    \[ Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1+N(s,a)} \]
    where:
    -   `Q(s,a)` is the average return from taking action `a`.
    -   `P(s,a)` is the prior probability of selecting `a`.
    -   `N(s)` is the visit count of the parent node.
    -   `N(s,a)` is the visit count for the action edge.

2.  **Expansion**: When a leaf node is reached, if it's not a terminal state, it is expanded. New child nodes are added to the tree. The agent uses **progressive widening**, where the number of children to expand (`m_allow`) is a function of the parent node's visit count: `m_allow = k0 + k1 * (N(s)^alpha)`. This ensures that promising nodes are explored more broadly.

3.  **Simulation & Evaluation**: From the newly expanded node, a "simulation" is performed. In this implementation, instead of a full random rollout, the value of a leaf node is bootstrapped using the potential function `phi`.
    -   **`phi(state)`**: This potential function estimates the "badness" of a state based on its `z_hat` (proxy for overload). It's calculated as the negative sum of squares of positive overload values. `v = -phi(state)`. The change in phi (`phi(next_state) - phi(state)`) is used as a shaped reward during the search.
    -   **Commit Evaluation**: When a `CommitRegulation` action is selected, the MCTS search calls `_evaluate_commit`. This invokes the `RateFinder` to determine the optimal rates and the resulting objective improvement (`delta_j`). The negative of this improvement (`-delta_j`) is used as a real reward for this action. This is the most computationally expensive part of the search.

4.  **Backpropagation (Backup)**: The return from the simulation (either the bootstrapped value `v` or the real reward `-delta_j` plus shaped rewards) is backed up the path of nodes visited during the selection phase. The `N` (visit count) and `W` (total value) statistics of the nodes and edges along the path are updated. `Q` is then `W/N`.

## 5. Logging

The `SearchLogger` provides a structured JSON-based logging mechanism to trace the agent's execution.

-   **Initialization**: It's typically configured to write to a timestamped file in a specified directory.
```python
from parrhesia.flow_agent import SearchLogger
log_dir = "path/to/logs"
logger = SearchLogger.to_timestamped(str(log_dir))
```
-   **Log Format**: Each line in the log file is a JSON object with a `type`, a `timestamp`, and a `payload` specific to the event type.

-   **Key Events**:
    -   `run_start`: Logged at the beginning of `MCTSAgent.run()`. The payload includes the number of hotspot candidates and the MCTS configuration.
    -   `after_commit`: Logged after a regulation is successfully found and committed. The payload contains the canonical dictionary of the regulation, the `delta_j` (objective improvement), and the current commit count.
    -   `mcts_error`: Logged if an exception occurs during `mcts.run()`.
    -   `regulation_limit_reached`: Logged if the agent stops because it has reached `max_regulations`.
    -   `run_end`: Logged at the end of `MCTSAgent.run()`. The payload contains the final commit count and the summary of the final objective evaluation, including `action_counts`.

**Example Log Output Snippets:**
```json
{"type": "run_start", "timestamp": "...", "payload": {"num_candidates": 5, "mcts_cfg": {...}}}
{"type": "after_commit", "timestamp": "...", "payload": {"reg": {...}, "delta_j": -150.7, "commits": 1}}
{"type": "run_end", "timestamp": "...", "payload": {"commits": 1, "objective": -150.7, "components": {...}, "action_counts": {"AddFlow": 20, "PickHotspot": 5, ...}}}
```

## 6. Inputs and Outputs

### Inputs

To run the `MCTSAgent`, you need to provide several key artifacts and configurations. These are typically loaded from files.

-   **Occupancy Matrix**: A sparse matrix representing which flights occupy which traffic volumes at which time bins. (e.g., `so6_occupancy_matrix_with_times.json`)
-   **TVTW Indexer**: A JSON file that maps traffic volume IDs and time information. (e.g., `tvtw_indexer.json`)
-   **Capacities GeoJSON**: A GeoJSON file defining the capacity for each traffic volume. (e.g., `wxm_sm_ih_maxpool.geojson`)

These artifacts are used to initialize the `FlightList` and `NetworkEvaluator`, which are then passed to the `MCTSAgent`.

### Outputs

The `MCTSAgent.run()` method returns a tuple `(final_state, run_info)`.

-   `final_state: PlanState`: The final state after the run. `final_state.plan` contains the list of `RegulationSpec` objects that were committed. Each `RegulationSpec` describes a single regulation, including the control volume, time window, flows, rates, and diagnostic information.
-   `run_info: RunInfo`: A dataclass containing metadata about the run.
    -   `commits: int`: Number of regulations committed.
    -   `total_delta_j: float`: Sum of objective improvements from all committed regulations.
    -   `log_path: str`: Path to the generated log file.
    -   `summary: dict`: A dictionary containing the final global objective score and its components.
    -   `action_counts: dict`: A dictionary counting how many times each type of action was taken during the search.

## 7. Example Usage

The following example, based on the project's test suite, demonstrates how to set up and run the `MCTSAgent`.

```python
# 1. Import necessary classes
from pathlib import Path
import geopandas as gpd
from parrhesia.flow_agent import (
    MCTSAgent,
    MCTSConfig,
    RateFinderConfig,
    HotspotDiscoveryConfig,
    SearchLogger,
)
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator

# 2. Define paths to artifact files
occupancy_path = Path("path/to/so6_occupancy_matrix_with_times.json")
indexer_path = Path("path/to/tvtw_indexer.json")
caps_path = Path("path/to/wxm_sm_ih_maxpool.geojson")

# 3. Load data and initialize core components
indexer = TVTWIndexer.load(str(indexer_path))
flight_list = FlightList(str(occupancy_path), str(indexer_path))
caps_gdf = gpd.read_file(str(caps_path))
evaluator = NetworkEvaluator(caps_gdf, flight_list)

# 4. Configure the agent
log_dir = Path("/tmp/mcts_runs")
logger = SearchLogger.to_timestamped(str(log_dir))

mcts_cfg = MCTSConfig(max_sims=12, commit_depth=1, commit_eval_limit=6, seed=0)
rf_cfg = RateFinderConfig(use_adaptive_grid=True, max_eval_calls=128)
disc_cfg = HotspotDiscoveryConfig(
    threshold=0.0,
    top_hotspots=5,
    top_flows=3,
    max_flights_per_flow=15,
    leiden_params={"threshold": 0.3, "resolution": 1.0, "seed": 0},
    direction_opts={"mode": "none"},
)

# 5. Initialize and run the agent
agent = MCTSAgent(
    evaluator=evaluator,
    flight_list=flight_list,
    indexer=indexer,
    mcts_cfg=mcts_cfg,
    rate_finder_cfg=rf_cfg,
    discovery_cfg=disc_cfg,
    logger=logger,
    max_regulations=1,
)

print("Starting agent.run()...")
final_state, run_info = agent.run()
logger.close()

# 6. Inspect results
print(f"Committed {run_info.commits} regulations.")
print(f"Final plan: {final_state.plan}")
print(f"Final objective summary: {run_info.summary}")
print(f"Log file at: {run_info.log_path}")
```
