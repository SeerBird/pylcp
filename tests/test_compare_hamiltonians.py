import os
import numpy as np
import pytest

import pylcp as fork
from tests.test_configs import (
    SINGLE_F_CONFIG, HYPERFINE_COUPLED_CONFIG, FINE_STRUCTURE_CONFIG,
    DQ_TWO_HYPERFINE_CONFIG, DQ_BARE_HYPERFINE_CONFIG,
    TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED, TWO_LEVEL_DQIJ, TWO_LEVEL_MASS
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_singleF_comparison():
    """Compare singleF Hamiltonian matrices H0 and mu_q against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "hamiltonians_ref.npz"))
    H0_fk, muq_fk = fork.hamiltonians.singleF(**SINGLE_F_CONFIG)

    np.testing.assert_allclose(H0_fk, ref["H0_sf"])
    np.testing.assert_allclose(muq_fk, ref["muq_sf"])


def test_hyperfine_coupled_comparison():
    """Compare hyperfine_coupled Hamiltonian matrices H0 and mu_q against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "hamiltonians_ref.npz"))
    H0_fk, muq_fk = fork.hamiltonians.hyperfine_coupled(**HYPERFINE_COUPLED_CONFIG)

    np.testing.assert_allclose(H0_fk, ref["H0_hf"])
    np.testing.assert_allclose(muq_fk, ref["muq_hf"])


def test_fine_structure_uncoupled_comparison():
    """Compare fine_structure_uncoupled Kronecker product optimization against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "hamiltonians_ref.npz"))
    H0_fk, muq_fk = fork.hamiltonians.fine_structure_uncoupled(**FINE_STRUCTURE_CONFIG)

    np.testing.assert_allclose(H0_fk, ref["H0_fs"], atol=1e-10)
    np.testing.assert_allclose(muq_fk, ref["muq_fs"], atol=1e-10)


def test_hyperfine_uncoupled_vectorized():
    """Test optimized hyperfine_uncoupled Kronecker product formulation in fork."""
    H0_fk, muq_fk = fork.hamiltonians.hyperfine_uncoupled(
        J=1.5, I=2.5, gJ=2.0, gI=-0.001, Ahfs=1e6
    )

    # 1. Dimension check (2J+1)*(2I+1) = 4 * 6 = 24 states
    assert H0_fk.shape == (24, 24)
    assert muq_fk.shape == (3, 24, 24)

    # 2. H0 Hermiticity
    np.testing.assert_allclose(H0_fk, H0_fk.T.conj(), atol=1e-10)


def test_dqij_norm_comparison():
    """Compare dqij_norm vectorization against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "hamiltonians_ref.npz"))
    norm_fk = fork.hamiltonians.dqij_norm(ref["dq_raw"])

    np.testing.assert_allclose(norm_fk, ref["dq_norm"], atol=1e-14)


def test_dqij_two_hyperfine_manifolds_comparison():
    """Compare dqij_two_hyperfine_manifolds against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "hamiltonians_ref.npz"))
    dq_fk = fork.hamiltonians.dqij_two_hyperfine_manifolds(**DQ_TWO_HYPERFINE_CONFIG)

    np.testing.assert_allclose(dq_fk, ref["dq_two_hf"], atol=1e-10)


def test_dqij_two_bare_hyperfine_comparison():
    """Compare dqij dipole coupling matrices against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "hamiltonians_ref.npz"))
    dq_fk = fork.hamiltonians.dqij_two_bare_hyperfine(**DQ_BARE_HYPERFINE_CONFIG)

    np.testing.assert_allclose(dq_fk, ref["dq_bare_hf"])


def test_hamiltonian_object_comparison():
    """Compare constructed Hamiltonian object properties against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "hamiltonians_ref.npz"))
    H0_g_fk, muq_g_fk = fork.hamiltonians.singleF(**TWO_LEVEL_GROUND)
    H0_e_fk, muq_e_fk = fork.hamiltonians.singleF(**TWO_LEVEL_EXCITED)
    dqij_fk = fork.hamiltonians.dqij_two_bare_hyperfine(**TWO_LEVEL_DQIJ)
    H_fk = fork.Hamiltonian(H0_g_fk, H0_e_fk, muq_g_fk, muq_e_fk, dqij_fk, mass=TWO_LEVEL_MASS)

    np.testing.assert_array_equal(H_fk.ns, ref["ham_ns"])
    assert H_fk.n == int(ref["ham_n"])
    assert len(H_fk.blocks) == int(ref["ham_blocks_len"])
