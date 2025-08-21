from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TabuConfig:
    # Objective weights
    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 0.1
    delta: float = 25.0

    # Tabu parameters
    tenure: int = 15
    max_iterations: int = 50
    no_improve_patience: int = 10

    # Neighborhood controls
    beam_width: int = 10
    candidate_pool_size: int = 200
    rate_step: int = 5
    max_regulations: int = 5

    # Randomness
    seed: int | None = 42


