import os
import numpy as np
import pytest

import pylcp as fork
from tests.test_configs import (
    TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED, TWO_LEVEL_DQIJ, TWO_LEVEL_MASS,
    OBE_MOT_BEAMS_CONFIG, ZERO_B_VECTOR,
    OBE_EVAL_R, OBE_EVAL_T,
    FORCE_GRID_R, FORCE_GRID_V
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def create_fork_obe_system():
    """Helper to create F_g=0 -> F_e=1 OBE system in fork."""
    H0_g_fk, muq_g_fk = fork.hamiltonians.singleF(**TWO_LEVEL_GROUND)
    H0_e_fk, muq_e_fk = fork.hamiltonians.singleF(**TWO_LEVEL_EXCITED)
    dqij_fk = fork.hamiltonians.dqij_two_bare_hyperfine(**TWO_LEVEL_DQIJ)
    H_fk = fork.Hamiltonian(H0_g_fk, H0_e_fk, muq_g_fk, muq_e_fk, dqij_fk, mass=TWO_LEVEL_MASS)
    beams_fk = fork.Conventional3DMOTBeams(**OBE_MOT_BEAMS_CONFIG)
    mag_fk = fork.ConstantMagneticField(ZERO_B_VECTOR)
    eq_fk = fork.OBE(beams_fk, mag_fk, H_fk, transform_into_re_im=False)
    return eq_fk


def test_obe_evolution_matrix_comparison():
    """Compare full OBE evolution matrices against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "obe_ref.npz"))
    eq_fk = create_fork_obe_system()

    ev_fk = eq_fk.full_OBE_ev(OBE_EVAL_R, OBE_EVAL_T)
    np.testing.assert_allclose(ev_fk, ref["ev_mat"], atol=1e-10)


def test_obe_force_profile_comparison():
    """Compare generated OBE force profiles against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "obe_ref.npz"))
    eq_fk = create_fork_obe_system()

    eq_fk.generate_force_profile(FORCE_GRID_R, FORCE_GRID_V, name="test_prof")
    prof_fk = eq_fk.profile["test_prof"]

    np.testing.assert_allclose(prof_fk.F, ref["force_prof_F"], atol=1e-8)
