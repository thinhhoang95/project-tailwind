"""Regulation Zero: MCTS-driven regulation search environment.

This package provides a self-contained environment and MCTS search implementation
for exploring regulation proposals ("regen") over a sandboxed copy of the
flight list. It integrates with existing modules for hotspot extraction,
flow computation, proposal generation, and plan evaluation.

Key modules:
- types: Small dataclasses and type aliases for actions, config, and path keys.
- cache: Lightweight node/child stats cache for MCTS and proposals memoization.
- env: RZSandbox wrapping AppResources with structural flight list cloning and
       safe bindings for global resources used by flows/evaluator.
- mcts: PUCT-based MCTS without rollouts using immediate regen deltas as rewards.
- runner: Glue utilities to create an env factory and run a search from CLI.
"""

__all__ = [
    "types",
    "cache",
    "env",
    "mcts",
    "runner",
]

