"""Measured robust design baseline for the current SKY130 CTLE topology.

The design below was sampled by ``experiments.feasibility_band`` and then re-evaluated
with the full (``fast=False``) guard and metric path on the exact 32 held-out target /
channel pairs. It is intentionally a baseline, not an RL result: it gives the product
an honest 32/32 safety net while the residual policy learns whether retargeting is useful.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from eqrl.circuits.ctle import DesignVars, encode_action


# Full-SKY130 candidate from results/robust_library_candidate1.json.
ROBUST_DESIGN = DesignVars(
    w_in=47.9002339241e-6,
    l_in=0.245544880270e-6,
    i_tail=0.623590074051e-3,
    rs=2448.0628249591614,
    cs=233.5542233175439e-15,
    r_load=3330.44753238441,
)


def robust_design() -> DesignVars:
    """Return a copy so callers cannot mutate the shared baseline fixture."""
    return dataclasses.replace(ROBUST_DESIGN)


def robust_unit_action() -> np.ndarray:
    """Return the baseline in the environment's normalized [0, 1] coordinates."""
    return encode_action(ROBUST_DESIGN)
