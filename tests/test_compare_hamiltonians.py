import numpy as np
import pytest
import tests.vendor.upstream_pylcp as upstream
import pylcp as fork

def test_singleF_comparison():
    """Compare singleF Hamiltonian matrices H0 and mu_q."""
    H0_up, muq_up = upstream.hamiltonians.singleF(F=1, gF=1.5)
    H0_fk, muq_fk = fork.hamiltonians.singleF(F=1, gF=1.5)
    
    np.testing.assert_allclose(H0_up, H0_fk)
    np.testing.assert_allclose(muq_up, muq_fk)

def test_hyperfine_coupled_comparison():
    """Compare hyperfine_coupled Hamiltonian matrices H0 and mu_q."""
    H0_up, muq_up = upstream.hamiltonians.hyperfine_coupled(J=0.5, I=1.5, gJ=2.0, gI=-0.000995, Ahfs=3.417e9)
    H0_fk, muq_fk = fork.hamiltonians.hyperfine_coupled(J=0.5, I=1.5, gJ=2.0, gI=-0.000995, Ahfs=3.417e9)
    
    np.testing.assert_allclose(H0_up, H0_fk)
    np.testing.assert_allclose(muq_up, muq_fk)

def test_fine_structure_uncoupled_comparison():
    """Compare fine_structure_uncoupled Kronecker product optimization against upstream."""
    H0_up, muq_up = upstream.hamiltonians.fine_structure_uncoupled(L=1, S=0.5, I=1.5, xi=1e9, a_c=1e6, a_orb=1e5, a_dip=1e4, gL=1.0, gS=2.0, gI=-0.001)
    H0_fk, muq_fk = fork.hamiltonians.fine_structure_uncoupled(L=1, S=0.5, I=1.5, xi=1e9, a_c=1e6, a_orb=1e5, a_dip=1e4, gL=1.0, gS=2.0, gI=-0.001)
    
    np.testing.assert_allclose(H0_up, H0_fk, atol=1e-10)
    np.testing.assert_allclose(muq_up, muq_fk, atol=1e-10)

def test_hyperfine_uncoupled_vectorized():
    """Test optimized hyperfine_uncoupled Kronecker product formulation in fork."""
    H0_fk, muq_fk = fork.hamiltonians.hyperfine_uncoupled(J=1.5, I=2.5, gJ=2.0, gI=-0.001, Ahfs=1e6)
    
    # 1. Dimension check (2J+1)*(2I+1) = 4 * 6 = 24 states
    assert H0_fk.shape == (24, 24)
    assert muq_fk.shape == (3, 24, 24)
    
    # 2. H0 Hermiticity
    np.testing.assert_allclose(H0_fk, H0_fk.T.conj(), atol=1e-10)

def test_dqij_norm_comparison():
    """Compare dqij_norm vectorization against upstream."""
    dq_raw = np.random.rand(3, 8, 12)
    norm_up = upstream.hamiltonians.dqij_norm(dq_raw)
    norm_fk = fork.hamiltonians.dqij_norm(dq_raw)
    
    np.testing.assert_allclose(norm_up, norm_fk, atol=1e-14)

def test_dqij_two_hyperfine_manifolds_comparison():
    """Compare dqij_two_hyperfine_manifolds between upstream and fork."""
    dq_up = upstream.hamiltonians.dqij_two_hyperfine_manifolds(J=0.5, Jp=1.5, I=1.5)
    dq_fk = fork.hamiltonians.dqij_two_hyperfine_manifolds(J=0.5, Jp=1.5, I=1.5)
    
    np.testing.assert_allclose(dq_up, dq_fk, atol=1e-10)

def test_dqij_two_bare_hyperfine_comparison():
    """Compare dqij dipole coupling matrices."""
    dq_up = upstream.hamiltonians.dqij_two_bare_hyperfine(F=1, Fp=2, normalize=True)
    dq_fk = fork.hamiltonians.dqij_two_bare_hyperfine(F=1, Fp=2, normalize=True)
    
    np.testing.assert_allclose(dq_up, dq_fk)

def test_hamiltonian_object_comparison():
    """Compare constructed hamiltonian object properties."""
    H0_g_up, muq_g_up = upstream.hamiltonians.singleF(F=0, gF=1.0)
    H0_e_up, muq_e_up = upstream.hamiltonians.singleF(F=1, gF=1.0)
    dqij_up = upstream.hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    H_up = upstream.hamiltonian(H0_g_up, H0_e_up, muq_g_up, muq_e_up, dqij_up, mass=87.0)

    H0_g_fk, muq_g_fk = fork.hamiltonians.singleF(F=0, gF=1.0)
    H0_e_fk, muq_e_fk = fork.hamiltonians.singleF(F=1, gF=1.0)
    dqij_fk = fork.hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    H_fk = fork.hamiltonian(H0_g_fk, H0_e_fk, muq_g_fk, muq_e_fk, dqij_fk, mass=87.0)

    np.testing.assert_array_equal(H_up.ns, H_fk.ns)
    assert H_up.n == H_fk.n
    assert len(H_up.blocks) == len(H_fk.blocks)
