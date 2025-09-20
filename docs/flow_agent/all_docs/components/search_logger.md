**Search Logger**

- Purpose: Lightweight JSONL logger to trace search runs and outcomes.
- Location: `src/parrhesia/flow_agent/logging.py`

Usage
- Creating a timestamped log file
```python
from parrhesia.flow_agent import SearchLogger
logger = SearchLogger.to_timestamped("output/flow_agent_runs")
```

- Writing events
```python
logger.event("run_start", {"num_candidates": 5, "mcts_cfg": {...}})
logger.event("after_commit", {"reg": {...}, "delta_j": -120.5, "commits": 1})
logger.event("run_end", {"commits": 1, "objective": -120.5, "components": {...}})
logger.close()
```

Format
- One JSON object per line with fields: `ts` (UTC ISO8601), `type`, and the event payload.
- The logger auto-serializes NumPy scalars/arrays and datetimes where possible; falls back to `str(obj)`.

Common events emitted by `MCTSAgent`
- `run_start`: At the beginning of a run, includes number of hotspot candidates and MCTS config.
- `after_commit`: After each committed regulation; includes canonical regulation dict and `delta_j`.
- `regulation_limit_reached`: Emitted if `max_regulations` bound stops the loop.
- `mcts_error`: Emitted when the MCTS search raises an exception.
- `run_end`: Final objective summary; may include aggregated `action_counts`.

Tips
- The log path is stored in `RunInfo.log_path` for convenient access by post-processing tools.

