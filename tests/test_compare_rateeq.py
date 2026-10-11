import os
import numpy as np
import pytest

import pylcp as fork
from tests.test_configs import (
    TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED, TWO_LEVEL_DQIJ, TWO_LEVEL_MASS,
    OBE_MOT_BEAMS_CONFIG, QUAD_B_GRADIENT,
    RATEEQ_EQUILIBRIUM_TEST_POINTS,
    FORCE_GRID_R, FORCE_GRID_V
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def create_fork_rateeq_system():
    """Helper to create F_g=0 -> F_e=1 rateeq system in fork."""
    H0_g_fk, muq_g_fk = fork.hamiltonians.singleF(**TWO_LEVEL_GROUND)
    H0_e_fk, muq_e_fk = fork.hamiltonians.singleF(**TWO_LEVEL_EXCITED)
    dqij_fk = fork.hamiltonians.dqij_two_bare_hyperfine(**TWO_LEVEL_DQIJ)
    H_fk = fork.Hamiltonian(H0_g_fk, H0_e_fk, muq_g_fk, muq_e_fk, dqij_fk, mass=TWO_LEVEL_MASS)
    beams_fk = fork.Conventional3DMOTBeams(**OBE_MOT_BEAMS_CONFIG)
    mag_fk = fork.QuadrupoleMagneticField(QUAD_B_GRADIENT)
    eq_fk = fork.RateEq(beams_fk, mag_fk, H_fk)
    return eq_fk


def test_rateeq_equilibrium_populations_comparison():
    """Compare rate equation equilibrium populations against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "rateeq_ref.npz"))
    eq_fk = create_fork_rateeq_system()

    neq_0_fk = eq_fk.equilibrium_populations(RATEEQ_EQUILIBRIUM_TEST_POINTS[0][0], RATEEQ_EQUILIBRIUM_TEST_POINTS[0][1], t=0)
    neq_1_fk = eq_fk.equilibrium_populations(RATEEQ_EQUILIBRIUM_TEST_POINTS[1][0], RATEEQ_EQUILIBRIUM_TEST_POINTS[1][1], t=0)

    np.testing.assert_allclose(neq_0_fk, ref["neq_0"], atol=1e-10)
    np.testing.assert_allclose(neq_1_fk, ref["neq_1"], atol=1e-10)


def test_rateeq_force_profile_comparison():
    """Compare generated rate equation force profiles against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "rateeq_ref.npz"))
    eq_fk = create_fork_rateeq_system()

    eq_fk.generate_force_profile(FORCE_GRID_R, FORCE_GRID_V, name="test_prof")
    prof_fk = eq_fk.profile["test_prof"]

    np.testing.assert_allclose(prof_fk.F, ref["force_prof_F"], atol=1e-10)
