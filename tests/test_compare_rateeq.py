import os
import numpy as np
import pytest

import pylcp as fork

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def create_fork_rateeq_system():
    """Helper to create F_g=0 -> F_e=1 rateeq system in fork."""
    H0_g_fk, muq_g_fk = fork.hamiltonians.singleF(F=0, gF=1.0)
    H0_e_fk, muq_e_fk = fork.hamiltonians.singleF(F=1, gF=1.0)
    dqij_fk = fork.hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    H_fk = fork.Hamiltonian(H0_g_fk, H0_e_fk, muq_g_fk, muq_e_fk, dqij_fk, mass=87.0)
    beams_fk = fork.Conventional3DMOTBeams(s=1.0, delta=-1.0)
    mag_fk = fork.QuadrupoleMagneticField(15.0)
    eq_fk = fork.RateEq(beams_fk, mag_fk, H_fk)
    return eq_fk


def test_rateeq_equilibrium_populations_comparison():
    """Compare rate equation equilibrium populations against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "rateeq_ref.npz"))
    eq_fk = create_fork_rateeq_system()

    neq_0_fk = eq_fk.equilibrium_populations(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), t=0)
    neq_1_fk = eq_fk.equilibrium_populations(np.array([0.1, -0.2, 0.3]), np.array([1.0, 0.0, -0.5]), t=0)

    np.testing.assert_allclose(neq_0_fk, ref["neq_0"], atol=1e-10)
    np.testing.assert_allclose(neq_1_fk, ref["neq_1"], atol=1e-10)


def test_rateeq_force_profile_comparison():
    """Compare generated rate equation force profiles against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "rateeq_ref.npz"))
    eq_fk = create_fork_rateeq_system()

    grid = np.linspace(-0.5, 0.5, 2)
    R = np.array([grid, np.zeros(2), np.zeros(2)])
    V = np.array([np.zeros(2), np.zeros(2), np.zeros(2)])

    eq_fk.generate_force_profile(R, V, name="test_prof")
    prof_fk = eq_fk.profile["test_prof"]

    np.testing.assert_allclose(prof_fk.F, ref["force_prof_F"], atol=1e-10)
