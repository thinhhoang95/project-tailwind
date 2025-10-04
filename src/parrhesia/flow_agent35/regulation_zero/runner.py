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
    """Build an env factory that forks from a prepared root sandbox.

    This factory pattern is used to efficiently create new, independent sandbox
    environments for parallel processing or simulations. Each created sandbox will
    be a copy of the fully loaded and prepared root sandbox, avoiding the cost
    of reloading resources for each new instance.
    """
    # Initialize the root sandbox environment with preloaded application resources
    # and the given configuration. Preloading all resources ensures that the
    # root sandbox is in a complete and ready state.
    root = RZSandbox(root_res.preload_all(), cfg=cfg)

    def factory() -> RZSandbox:
        """Create a new sandbox instance by forking the root sandbox."""
        # The fork method creates a deep copy or an efficient copy-on-write clone
        # of the root sandbox, allowing for independent state modifications in the
        # new instance.
        return root.fork()

    # Return the factory function, which can be called to get new sandbox instances.
    return factory


def run_search(cfg: RZConfig) -> List[RZAction]:
    """Run MCTS over a fresh local AppResources and return an action sequence.

    This is the main entry point for executing a Regulation Zero search. It sets up
    the necessary resources and configuration, then invokes the MCTS algorithm
    to find an optimal sequence of actions.
    """
    # Instantiate AppResources and preload all necessary data. This might include
    # loading data from databases, configuration files, etc., into memory.
    res = AppResources().preload_all()

    # Create an MCTS instance. It is configured with a factory for creating
    # simulation environments and the search configuration object.
    mcts = MCTS(env_factory=make_env_factory(res, cfg), cfg=cfg)

    # Execute the MCTS algorithm. The run method will perform the search and
    # return the best sequence of actions found.
    return mcts.run()


# Expose the public functions of this module.
__all__ = ["make_env_factory", "run_search"]

