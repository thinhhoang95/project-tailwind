I would like you to help me implement this optimization framework based on Adaptive Large Neighborhood Search (or ALNS).

Below is the detailed implementation plan.

### Revised ALNS Implementation Plan

My goal is to refactor `ProblemState` to become the core state representation for the ALNS solver and then build the orchestrator around it.

**Part 1: Refactor `ProblemState` (`pstate.py`)**

I will modify the existing `ProblemState` class to represent a solution candidate (`NetworkPlan`) and the context required for its evaluation, making it compatible with the `alns` library.

*   **File to Edit**: `src/project_tailwind/optimize/alns/pstate.py`
*   **Class Refactoring**:
    1.  **Change Attributes**: The `ProblemState` will no longer store the `flight_list`. Instead, it will hold the decision variables.
        *   **Remove**: `flight_list`, `move_number`.
        *   **Add**:
            *   `network_plan`: A `NetworkPlan` object representing the solution.
            *   `optimization_problem`: A reference to the main `OptimizationProblem` instance, which contains the evaluation logic and all necessary data (baseline flights, traffic volumes, etc.).
    2.  **Add `objective()` method**: To make `ProblemState` compatible with the `alns` library's requirement for a state object, I will add an `objective()` method. This method will simply delegate the call to the `optimization_problem` instance, like so: `return self.optimization_problem.objective(self)`.
    3.  **Update `copy()` method**: The `copy()` method will be updated to create a new `ProblemState` with a copy of the `network_plan`. The reference to `optimization_problem` will be passed as-is, as it represents shared context.
    4.  **Remove `with_flight_list()`**: This method will no longer be relevant.

**Part 2: Adapt `OptimizationProblem` (`optimization_problem.py`)**

The `OptimizationProblem` class will be adjusted to work with the newly refactored `ProblemState`. It will act as the primary evaluator.

*   **File to Edit**: `src/project_tailwind/optimize/alns/optimization_problem.py`
*   **Class Refactoring**:
    1.  **Update `__init__`**: The initializer will be updated to accept and store the `RegulationParser` and `TVTWIndexer`, making it a self-contained context for evaluation.
    2.  **Update `create_initial_state()`**: This method will now instantiate `ProblemState` with an empty `NetworkPlan` and a reference to `self` (the `OptimizationProblem` instance).
    3.  **Rewrite `objective()` method**: This is the most critical change. The method will no longer receive a state with a pre-computed `flight_list`. Instead, it will perform the evaluation from scratch based on the `network_plan` in the passed state:
        *   Get the `network_plan` from the `state` object.
        *   Create a deep copy of the **base** `flight_list` (stored within `self`).
        *   Instantiate and apply a `NetworkPlanMove` to the copied flight list to simulate delays.
        *   Use `NetworkEvaluator` to compute the metrics (`z_95`, `z_sum`, delay) on the modified flight list.
        *   Calculate and return the final weighted objective score.

**Part 3: Implement ALNS Operators and Orchestrator**

This part of the plan remains largely the same, but it will now use the refactored `ProblemState`.

*   **New File**: `src/project_tailwind/optimize/alns/orchestrator.py`
*   **Destroy/Repair Operators**: I will implement `remove_regulation` and `add_regulation` as planned. They will now take the refactored `ProblemState` as input and return a new, modified `ProblemState`.
*   **Orchestrator Logic**:
    *   Set up and load all data as seen in `tests/test_alns_orchestra.py`.
    *   Instantiate the self-contained `OptimizationProblem`.
    *   Create the `initial_state` using `optimization_problem.create_initial_state()`.
    *   Configure and run the `alns` solver with the `RouletteWheel` selector, `HillClimbing` acceptance, and `MaxIterations` stopping criterion.
    *   Print the best solution found.

This approach correctly integrates the ALNS implementation into your existing project structure by adapting `ProblemState` and `OptimizationProblem`, which is a much cleaner solution.

