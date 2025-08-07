from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ProblemState:
    """
    Container for the optimization problem state used by ALNS.

    Requirements:
    - Moves must return a (deep) copy of the state instead of mutating the original.
    - We store the flight_list (the evolving plan) and auxiliary metadata.

    Attributes:
        flight_list: The current flight list object representing the plan/schedule.
        move_number: Integer counter of how many moves have been applied to reach this state.
        aux: Optional dictionary for extra, derived values that moves/operators may cache.
    """

    flight_list: Any
    move_number: int = 0
    aux: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "ProblemState":
        """
        Return a deep copy of the state, as required by ALNS move semantics.

        Notes:
        - We use deepcopy to ensure nested structures inside flight_list are copied.
        - If flight_list implements its own lightweight copy/clone, you can replace
          the deepcopy(flight_list) with that method to improve performance.
        """
        # Deep copy flight_list and aux to ensure isolation between states.
        copied_flight_list = deepcopy(self.flight_list)
        copied_aux = deepcopy(self.aux)
        return ProblemState(
            flight_list=copied_flight_list,
            move_number=self.move_number,
            aux=copied_aux,
        )

    def increment_move(self) -> "ProblemState":
        """
        Increment the move counter. Returns self for chaining.
        """
        self.move_number += 1
        return self

    def with_flight_list(self, flight_list: Any) -> "ProblemState":
        """
        Return a new state with a replaced flight_list but same move_number and aux.
        """
        new_state = self.copy()
        new_state.flight_list = flight_list
        return new_state

    def __repr__(self) -> str:
        fl_summary = getattr(self.flight_list, "summary", None)
        if callable(fl_summary):
            try:
                desc = fl_summary()
            except Exception:
                desc = f"{type(self.flight_list).__name__}"
        else:
            desc = f"{type(self.flight_list).__name__}"
        return f"ProblemState(move_number={self.move_number}, flight_list={desc}, aux_keys={list(self.aux.keys())})"
