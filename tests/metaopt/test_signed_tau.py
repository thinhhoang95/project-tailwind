from __future__ import annotations

import numpy as np

from pathlib import Path
import sys

# Ensure project src is importable
project_root = Path(__file__).parent.parent / ".." / "src"
sys.path.insert(0, str(project_root.resolve()))

from parrhesia.metaopt.travel_offsets import flow_offsets_from_ctrl
from parrhesia.metaopt.per_flow_features import phase_time
from parrhesia.metaopt.types import Hotspot


class _StubFlightList:
    def __init__(self, tv_id_to_idx: dict[str, int], seqs: dict[str, list[int]]):
        self.tv_id_to_idx = tv_id_to_idx
        self.idx_to_tv_id = {int(v): str(k) for k, v in tv_id_to_idx.items()}
        self._seqs = {str(k): np.asarray(v, dtype=np.int64) for k, v in seqs.items()}

    def get_flight_tv_sequence_indices(self, flight_id: str):
        arr = self._seqs.get(str(flight_id))
        if arr is None:
            return np.empty(0, dtype=np.int64)
        return arr


def test_flow_offsets_from_ctrl_signed_and_unsigned():
    # TVs and rows
    tv_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    # Bin offsets in minutes->bins already converted; provide only B->A and B->C
    bin_offsets = {
        "B": {"A": 1, "C": 2},
        # Reverse entries to exercise symmetry are optional
        # "A": {"B": 1},
        # "C": {"B": 2},
    }
    # Three flights: majority order A -> B -> C, so relative to control B:
    #  - A upstream (−)
    #  - C downstream (+)
    seqs = {
        "f1": [0, 1, 2],  # A,B,C
        "f2": [0, 1],     # A,B
        "f3": [1, 2],     # B,C
    }
    fl = _StubFlightList(tv_map, seqs)

    # Back-compat (no context): non-negative magnitudes
    tau_unsigned = flow_offsets_from_ctrl("B", tv_map, bin_offsets)
    assert tau_unsigned is not None
    assert tau_unsigned.get(1) == 0  # control row
    assert tau_unsigned.get(0) == 1  # |τ_{B->A}|
    assert tau_unsigned.get(2) == 2  # |τ_{B->C}|

    # Signed (order_vs_ctrl)
    tau_signed = flow_offsets_from_ctrl(
        "B",
        tv_map,
        bin_offsets,
        flow_flight_ids=["f1", "f2", "f3"],
        flight_list=fl,
        direction_sign_mode="order_vs_ctrl",
    )
    assert tau_signed is not None
    assert tau_signed.get(1) == 0  # control row
    # Majority places A before B -> upstream -> negative sign
    assert tau_signed.get(0) == -1
    # Majority places C after B -> downstream -> positive sign
    assert tau_signed.get(2) == 2


def test_phase_time_uses_hotspot_row_tau():
    # Given τ with hotspot row offset, ensure t_G = t* - τ_{G,s*}
    tv_map = {"A": 0, "B": 1, "C": 2}
    # Suppose control is B, τ magnitudes are 1 (A) and 2 (C), with signs A: -1, C: +2
    tau = {0: -1, 1: 0, 2: 2}
    T = 100

    # Hotspot at C (row 2), t* = 10 -> t_G = 10 - τ_{G,C} = 10 - 2 = 8
    h = Hotspot(tv_id="C", bin=10)
    tG = phase_time(2, h, tau, T)
    assert tG == 8

    # Hotspot at A (row 0), t* = 10 -> t_G = 10 - (-1) = 11
    h2 = Hotspot(tv_id="A", bin=10)
    tG2 = phase_time(0, h2, tau, T)
    assert tG2 == 11
