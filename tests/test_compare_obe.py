import os
import numpy as np
import pytest

import pylcp as fork

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def create_fork_obe_system():
    """Helper to create F_g=0 -> F_e=1 OBE system in fork."""
    H0_g_fk, muq_g_fk = fork.hamiltonians.singleF(F=0, gF=1.0)
    H0_e_fk, muq_e_fk = fork.hamiltonians.singleF(F=1, gF=1.0)
    dqij_fk = fork.hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    H_fk = fork.Hamiltonian(H0_g_fk, H0_e_fk, muq_g_fk, muq_e_fk, dqij_fk, mass=87.0)
    beams_fk = fork.Conventional3DMOTBeams(s=1.0, delta=-1.0)
    mag_fk = fork.ConstantMagneticField(np.array([0.0, 0.0, 0.0]))
    eq_fk = fork.OBE(beams_fk, mag_fk, H_fk, transform_into_re_im=False)
    return eq_fk


def test_obe_evolution_matrix_comparison():
    """Compare full OBE evolution matrices against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "obe_ref.npz"))
    eq_fk = create_fork_obe_system()

    r = np.array([0.1, -0.2, 0.3])
    t = 0.0

    ev_fk = eq_fk.full_OBE_ev(r, t)
    np.testing.assert_allclose(ev_fk, ref["ev_mat"], atol=1e-10)


def test_obe_force_profile_comparison():
    """Compare generated OBE force profiles against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "obe_ref.npz"))
    eq_fk = create_fork_obe_system()

    grid = np.linspace(-0.5, 0.5, 2)
    R = np.array([grid, np.zeros(2), np.zeros(2)])
    V = np.array([np.zeros(2), np.zeros(2), np.zeros(2)])

    eq_fk.generate_force_profile(R, V, name="test_prof")
    prof_fk = eq_fk.profile["test_prof"]

    np.testing.assert_allclose(prof_fk.F, ref["force_prof_F"], atol=1e-8)
