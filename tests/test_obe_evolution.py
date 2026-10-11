import os
import time
import numpy as np
import pytest

import pylcp as fork
from tests.test_configs import (
    TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED, TWO_LEVEL_DQIJ, TWO_LEVEL_MASS,
    ZERO_B_VECTOR,
    OBE_RABI_BEAM_CONFIG,
    OBE_MOTION_BEAM_CONFIG
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def create_fork_two_level_hamiltonian():
    """Helper to create two-level test Hamiltonian from shared configs."""
    H0_g, muq_g = fork.hamiltonians.singleF(**TWO_LEVEL_GROUND)
    H0_e, muq_e = fork.hamiltonians.singleF(**TWO_LEVEL_EXCITED)
    dqij = fork.hamiltonians.dqij_two_bare_hyperfine(**TWO_LEVEL_DQIJ)
    return fork.Hamiltonian(H0_g, H0_e, muq_g, muq_e, dqij, mass=TWO_LEVEL_MASS)


def test_obe_evolve_density_against_reference():
    """Verify fork OBE density matrix time evolution against precomputed upstream baseline."""
    ref_path = os.path.join(DATA_DIR, "obe_density_ref.npz")
    assert os.path.exists(ref_path), f"Reference benchmark not found at {ref_path}"
    ref = np.load(ref_path)

    H = create_fork_two_level_hamiltonian()
    beam = fork.LaserBeam(**OBE_RABI_BEAM_CONFIG)
    beams = fork.LaserBeams([beam])
    mag = fork.ConstantMagneticField(ZERO_B_VECTOR)

    obe = fork.OBE(beams, mag, H, transform_into_re_im=True)
    obe.set_initial_rho_equally()

    t0 = time.perf_counter()
    sol = obe.evolve_density([ref["t"][0], ref["t"][-1]], t_eval=ref["t"])
    t_fork = time.perf_counter() - t0

    np.testing.assert_allclose(sol.rho, ref["rho"], atol=1e-9)
    # Fork should execute under 1 second
    assert t_fork < 1.0


def test_obe_evolve_motion_against_reference():
    """Verify fork OBE coupled motion/density evolution against precomputed upstream baseline."""
    ref_path = os.path.join(DATA_DIR, "obe_motion_ref.npz")
    assert os.path.exists(ref_path), f"Reference benchmark not found at {ref_path}"
    ref = np.load(ref_path)

    H = create_fork_two_level_hamiltonian()
    beam = fork.LaserBeam(**OBE_MOTION_BEAM_CONFIG)
    beams = fork.LaserBeams([beam])
    mag = fork.ConstantMagneticField(ZERO_B_VECTOR)

    obe = fork.OBE(beams, mag, H, transform_into_re_im=True)
    obe.set_initial_position_and_velocity(ref["r0"], ref["v0"])
    obe.set_initial_rho_equally()

    t0 = time.perf_counter()
    sol = obe.evolve_motion([ref["t"][0], ref["t"][-1]], t_eval=ref["t"])
    t_fork = time.perf_counter() - t0

    np.testing.assert_allclose(sol.r, ref["r"], atol=1e-9)
    np.testing.assert_allclose(sol.v, ref["v"], atol=1e-9)
    np.testing.assert_allclose(sol.rho, ref["rho"], atol=1e-9)
    # Fork should execute under 1 second
    assert t_fork < 1.0


def test_obe_probability_conservation_and_steady_state():
    """Verify trace preservation and saturation steady state for resonant drive."""
    H = create_fork_two_level_hamiltonian()

    s = OBE_RABI_BEAM_CONFIG["s"]
    beam = fork.LaserBeam(**OBE_RABI_BEAM_CONFIG)
    beams = fork.LaserBeams([beam])
    mag = fork.ConstantMagneticField(ZERO_B_VECTOR)

    obe = fork.OBE(beams, mag, H, transform_into_re_im=True)
    obe.set_initial_rho_equally()

    t_eval = np.linspace(0, 15.0, 31)
    sol = obe.evolve_density([0, 15.0], t_eval=t_eval)

    # 1. Total population trace must equal 1.0 at all times
    # sol.rho has shape (N, N, t_points)
    total_trace = np.sum([sol.rho[ii, ii, :] for ii in range(H.n)], axis=0)
    np.testing.assert_allclose(total_trace, np.ones_like(t_eval), atol=1e-6)

    # 2. Excited state population at t=15 (steady state) should match s/2 / (1 + s)
    # For pi polarized beam addressing q=0, ground (state 0) couples to excited state 2 (m_e=0)
    rho_excited_total = np.sum([sol.rho[ii, ii, -1] for ii in range(1, 4)])
    expected_steady_excited = (s / 2.0) / (1.0 + s)
    assert np.isclose(rho_excited_total, expected_steady_excited, atol=1e-2)
