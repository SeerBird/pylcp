import os
import time
import numpy as np
import pytest

import pylcp as fork

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_rateeq_evolve_populations_against_reference():
    """Verify fork RateEq population relaxation against precomputed upstream baseline."""
    ref_path = os.path.join(DATA_DIR, "rateeq_populations_ref.npz")
    assert os.path.exists(ref_path), f"Reference benchmark not found at {ref_path}"
    ref = np.load(ref_path)

    H0_g, muq_g = fork.hamiltonians.singleF(F=0, gF=1.0)
    H0_e, muq_e = fork.hamiltonians.singleF(F=1, gF=1.0)
    dqij = fork.hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    H = fork.Hamiltonian(H0_g, H0_e, muq_g, muq_e, dqij, mass=87.0)
    beams = fork.Conventional3DMOTBeams(s=1.0, delta=-1.0)
    mag = fork.QuadrupoleMagneticField(15.0)

    eq = fork.RateEq(beams, mag, H)
    eq.set_initial_position_and_velocity(ref["r0"], ref["v0"])
    eq.set_initial_pop(ref["N0"])

    t0 = time.perf_counter()
    sol = eq.evolve_populations([ref["t"][0], ref["t"][-1]], t_eval=ref["t"])
    t_fork = time.perf_counter() - t0

    np.testing.assert_allclose(sol.y, ref["y"], atol=1e-9)
    # Execution should be very fast (< 0.5s)
    assert t_fork < 0.5


def test_rateeq_evolve_motion_against_reference():
    """Verify fork RateEq 3D MOT motion evolution against precomputed upstream baseline."""
    ref_path = os.path.join(DATA_DIR, "rateeq_motion_ref.npz")
    assert os.path.exists(ref_path), f"Reference benchmark not found at {ref_path}"
    ref = np.load(ref_path)

    H0_g, muq_g = fork.hamiltonians.singleF(F=0, gF=1.0)
    H0_e, muq_e = fork.hamiltonians.singleF(F=1, gF=1.0)
    dqij = fork.hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    H = fork.Hamiltonian(H0_g, H0_e, muq_g, muq_e, dqij, mass=87.0)
    beams = fork.Conventional3DMOTBeams(s=1.0, delta=-1.0)
    mag = fork.QuadrupoleMagneticField(15.0)

    eq = fork.RateEq(beams, mag, H)
    eq.set_initial_position_and_velocity(ref["r0"], ref["v0"])
    eq.set_initial_pop_from_equilibrium()

    t0 = time.perf_counter()
    sol = eq.evolve_motion([ref["t"][0], ref["t"][-1]], t_eval=ref["t"])
    t_fork = time.perf_counter() - t0

    np.testing.assert_allclose(sol.r, ref["r"], atol=1e-9)
    np.testing.assert_allclose(sol.v, ref["v"], atol=1e-9)
    np.testing.assert_allclose(sol.N, ref["N"], atol=1e-9)
    # Execution should be very fast (< 0.5s)
    assert t_fork < 0.5
