"""
Shared simulation and test configurations for PyLCP benchmarks and test suites.

This module serves as the single source of truth for physical parameters,
laser configurations, magnetic field configurations, and integration time grids.
Both `tests/generate_benchmarks.py` and the `tests/test_*.py` test files import
from here to ensure that benchmark generation and testing remain in lockstep.
"""

import numpy as np

from pylcp.typing import PolarizationEnum

# ==============================================================================
# 1. Standard 2-Level System (F_g = 0 -> F_e = 1)
# ==============================================================================
TWO_LEVEL_GROUND = {"F": 0, "gF": 1.0}
TWO_LEVEL_EXCITED = {"F": 1, "gF": 1.0}
TWO_LEVEL_DQIJ = {"F": 0, "Fp": 1, "normalize": True}
TWO_LEVEL_MASS = 87.0

# ==============================================================================
# 2. Atom Configurations
# ==============================================================================
ATOM_STATE_CONFIG = {
    "n": 5, "S": 0.5, "L": 1, "J": 1.5,
    "lam": 780e-9, "tau": 26e-9, "gJ": 1.33
}
ATOM_RB_SPECIES = "87Rb"

# ==============================================================================
# 3. Magnetic Fields & Beams
# ==============================================================================
CONST_B_VECTOR = np.array([1.2, -0.5, 3.4])
CONST_B_POS = np.array([0.5, -1.0, 2.0])

QUAD_B_GRADIENT = 15.0
QUAD_B_POSITIONS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([1.0, -2.0, 3.0]),
    np.array([-0.5, 0.5, -1.0])
]

MOT_BEAMS_CONFIG = {"s": 2.0, "delta": -1.5}

# Zero field for testing isotropic/isolated setups
ZERO_B_VECTOR = np.array([0.0, 0.0, 0.0])

# ==============================================================================
# 4. Hamiltonian Sub-methods
# ==============================================================================
SINGLE_F_CONFIG = {"F": 1, "gF": 1.5}
HYPERFINE_COUPLED_CONFIG = {
    "J": 0.5, "I": 1.5, "gJ": 2.0, "gI": -0.000995, "Ahfs": 3.417e9
}
FINE_STRUCTURE_CONFIG = {
    "L": 1, "S": 0.5, "I": 1.5, "xi": 1e9, "a_c": 1e6, "a_orb": 1e5,
    "a_dip": 1e4, "gL": 1.0, "gS": 2.0, "gI": -0.001
}
DQ_TWO_HYPERFINE_CONFIG = {"J": 0.5, "Jp": 1.5, "I": 1.5}
DQ_BARE_HYPERFINE_CONFIG = {"F": 1, "Fp": 2, "normalize": True}

# ==============================================================================
# 5. OBE Setups & Grids
# ==============================================================================
OBE_MOT_BEAMS_CONFIG = {"s": 1.0, "delta": -1.0}
OBE_EVAL_R = np.array([0.1, -0.2, 0.3])
OBE_EVAL_T = 0.0

# 1D Force profile evaluation grid
FORCE_GRID_1D = np.linspace(-0.5, 0.5, 2)
FORCE_GRID_R = np.array([FORCE_GRID_1D, np.zeros(2), np.zeros(2)])
FORCE_GRID_V = np.array([np.zeros(2), np.zeros(2), np.zeros(2)])

# OBE Density Evolution (Rabi oscillations + spontaneous emission)
OBE_RABI_BEAM_CONFIG = {
    "kvec": np.array([0, 0, 1]), "pol": np.array([0, 0, 1]),
    "s": 2.0, "delta": 0.0
}
OBE_DENSITY_TIME = {
    "t_span": [0.0, 5.0],
    "t_eval": np.linspace(0.0, 5.0, 51)
}

# OBE Motion Evolution (Coupled trajectory)
OBE_MOTION_BEAM_CONFIG = {
    "kvec": np.array([0, 0, 1]), "pol": PolarizationEnum.SIGMA_PLUS,
    "s": 1.0, "delta": -1.0
}
OBE_MOTION_R0 = np.array([0.0, 0.0, 0.0])
OBE_MOTION_V0 = np.array([0.0, 0.0, 0.5])
OBE_MOTION_TIME = {
    "t_span": [0.0, 0.05],
    "t_eval": np.linspace(0.0, 0.05, 6)
}

# ==============================================================================
# 6. RateEq Setups
# ==============================================================================
RATEEQ_EQUILIBRIUM_TEST_POINTS = [
    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
    (np.array([0.1, -0.2, 0.3]), np.array([1.0, 0.0, -0.5])),
]

# RateEq Population relaxation
RATEEQ_POP_R0 = np.array([0.1, -0.2, 0.3])
RATEEQ_POP_V0 = np.array([0.5, 0.0, -0.2])
RATEEQ_POP_N0 = np.array([1.0, 0.0, 0.0, 0.0])
RATEEQ_POP_TIME = {
    "t_span": [0.0, 5.0],
    "t_eval": np.linspace(0.0, 5.0, 26)
}

# RateEq Motion Evolution (3D MOT trajectory)
RATEEQ_MOTION_R0 = np.array([0.2, 0.1, -0.1])
RATEEQ_MOTION_V0 = np.array([1.0, -0.5, 0.2])
RATEEQ_MOTION_TIME = {
    "t_span": [0.0, 2.0],
    "t_eval": np.linspace(0.0, 2.0, 21)
}
