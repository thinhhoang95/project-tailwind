from __future__ import annotations

"""
Run a Regulation Zero MCTS search and apply the chosen actions.

This script:
- Loads local AppResources (preloaded for isolation from any server instance).
- Runs MCTS for a small number of simulations (tunable) with a specified depth.
- Prints the selected action sequence.
- Applies up to `commit_depth` actions to a fresh sandbox to materialize a
  concrete solution, asserting state transitions work end-to-end.

Usage (from repo root):
    /Users/thinhhoang/miniforge3/envs/silverdrizzle/bin/python examples/regulation_zero/run_search.py \
        --sims 2 --depth 3 --commit-depth 3
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


# Ensure 'src' is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from server_tailwind.core.resources import AppResources

from parrhesia.flow_agent35.regulation_zero.types import RZAction, RZConfig, RZPathKey
from parrhesia.flow_agent35.regulation_zero.env import RZSandbox
from parrhesia.flow_agent35.regulation_zero.mcts import MCTS


def _make_env_factory(res: AppResources, cfg: RZConfig):
    root = RZSandbox(res.preload_all(), cfg=cfg)

    def factory() -> RZSandbox:
        return root.fork()

    return factory


def _greedy_path_by_visits(tt) -> List[Tuple[str, int]]:
    """Derive action keys (hotspot_key, rank) greedily by child visit counts."""
    path: List[Tuple[str, int]] = []
    state: RZPathKey = ()
    while state in tt and tt[state].children:
        ak, _cs = max(tt[state].children.items(), key=lambda kv: kv[1].N)
        path.append(ak)
        state = tuple(list(state) + [ak])
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run MCTS and apply top actions in a sandbox")
    ap.add_argument("--sims", type=int, default=2, help="number of MCTS simulations (small for smoke)")
    ap.add_argument("--depth", type=int, default=3, help="max MCTS depth")
    ap.add_argument("--commit-depth", type=int, default=3, help="number of actions to apply after search")
    args = ap.parse_args()

    cfg = RZConfig(
        max_depth=int(args.depth),
        num_simulations=int(args.sims),
    )

    print(f"[rz] Loading AppResources (isolated) ...")
    res = AppResources().preload_all()

    # Run MCTS
    print(f"[rz] Running MCTS: sims={cfg.num_simulations}, depth={cfg.max_depth}")
    mcts = MCTS(env_factory=_make_env_factory(res, cfg), cfg=cfg)
    _ = mcts.run()  # we will re-derive the greedy path from the transposition table
    chosen_keys = _greedy_path_by_visits(mcts.tt)
    if not chosen_keys:
        print("[rz] No actions selected by MCTS (empty tree).")
        return

    print("[rz] Chosen action sequence (hotspot_key, proposal_rank):")
    for i, (hk, r) in enumerate(chosen_keys, start=1):
        print(f"  {i:2d}. {hk} -> rank {r}")

    # Apply up to commit_depth actions on a fresh sandbox from baseline
    k_commit = max(0, min(int(args.commit_depth), len(chosen_keys)))
    print(f"[rz] Applying top {k_commit} actions to a fresh sandbox ...")
    env = _make_env_factory(res, cfg)()
    state: RZPathKey = ()
    total_delta = 0.0
    for i, (hk, rank) in enumerate(chosen_keys[:k_commit], start=1):
        key = (state, hk)
        ranked = mcts.cache.get(key)
        if not ranked:
            raise RuntimeError(f"Cache miss at step {i} for key={key}")
        if not (0 <= int(rank) < len(ranked)):
            raise IndexError(f"Rank out of range at step {i}: rank={rank}, len={len(ranked)}")
        prop, f2f, delta = ranked[int(rank)]
        print(f"  - Step {i}: {hk} rank={rank}  delta={float(delta):.3f}")
        env.apply_proposal(prop, f2f)
        total_delta += float(delta)
        # advance canonical state
        state = tuple(list(state) + [(hk, int(rank))])

    print(f"[rz] Applied {k_commit} regulations; cumulative predicted delta={total_delta:.3f}")
    fl = env.resources.flight_list
    print(f"[rz] Sandbox now: num_regulations={fl.num_regulations}, total_delay_assigned_min={fl.total_delay_assigned_min}")


if __name__ == "__main__":
    main()

