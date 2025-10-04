from __future__ import annotations

"""Runner glue for Regulation Zero search from a CLI or notebook.

Constructs a local AppResources instance, creates an RZSandbox, and runs MCTS
to produce an ordered list of actions (hotspot → proposal_rank pairs).
"""

from typing import Callable, List

from server_tailwind.core.resources import AppResources

from .env import RZSandbox
from .mcts import MCTS
from .types import RZAction, RZConfig


def make_env_factory(root_res: AppResources, cfg: RZConfig) -> Callable[[], RZSandbox]:
    """Build an env factory that forks from a prepared root sandbox."""
    root = RZSandbox(root_res.preload_all(), cfg=cfg)

    def factory() -> RZSandbox:
        return root.fork()

    return factory


def run_search(cfg: RZConfig) -> List[RZAction]:
    """Run MCTS over a fresh local AppResources and return an action sequence."""
    res = AppResources().preload_all()
    mcts = MCTS(env_factory=make_env_factory(res, cfg), cfg=cfg)
    return mcts.run()


__all__ = ["make_env_factory", "run_search"]

