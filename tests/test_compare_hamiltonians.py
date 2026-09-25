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
