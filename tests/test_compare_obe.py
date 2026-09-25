import numpy as np
import pytest
import tests.vendor.upstream_pylcp as upstream
import pylcp as fork

def create_obe_systems():
    """Helper to create matching F_g=0 -> F_e=1 OBE systems in upstream and fork."""
    # Upstream
    H0_g_up, muq_g_up = upstream.hamiltonians.singleF(F=0, gF=1.0)
    H0_e_up, muq_e_up = upstream.hamiltonians.singleF(F=1, gF=1.0)
    dqij_up = upstream.hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    H_up = upstream.hamiltonian(H0_g_up, H0_e_up, muq_g_up, muq_e_up, dqij_up, mass=87.0)
    beams_up = upstream.conventional3DMOTBeams(s=1.0, delta=-1.0)
    mag_up = upstream.constantMagneticField(np.array([0.0, 0.0, 0.0]))
    eq_up = upstream.obe(beams_up, mag_up, H_up, transform_into_re_im=False)

    # Fork
    H0_g_fk, muq_g_fk = fork.hamiltonians.singleF(F=0, gF=1.0)
    H0_e_fk, muq_e_fk = fork.hamiltonians.singleF(F=1, gF=1.0)
    dqij_fk = fork.hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    H_fk = fork.hamiltonian(H0_g_fk, H0_e_fk, muq_g_fk, muq_e_fk, dqij_fk, mass=87.0)
    beams_fk = fork.conventional3DMOTBeams(s=1.0, delta=-1.0)
    mag_fk = fork.constantMagneticField(np.array([0.0, 0.0, 0.0]))
    eq_fk = fork.obe(beams_fk, mag_fk, H_fk, transform_into_re_im=False)

    return eq_up, eq_fk

def test_obe_evolution_matrix_comparison():
    """Compare full OBE evolution matrices computed at position r and t=0."""
    eq_up, eq_fk = create_obe_systems()
    
    r = np.array([0.1, -0.2, 0.3])
    t = 0.0
    
    ev_up = eq_up.full_OBE_ev(r, t)
    ev_fk = eq_fk.full_OBE_ev(r, t)
    
    np.testing.assert_allclose(ev_up, ev_fk, atol=1e-10)

def test_obe_force_profile_comparison():
    """Compare generated OBE force profiles between upstream and fork."""
    eq_up, eq_fk = create_obe_systems()
    
    grid = np.linspace(-1, 1, 5)
    R = np.array([grid, np.zeros(5), np.zeros(5)])
    V = np.array([np.zeros(5), np.zeros(5), np.zeros(5)])
    
    eq_up.generate_force_profile(R, V, name='test_prof')
    eq_fk.generate_force_profile(R, V, name='test_prof')
    
    prof_up = eq_up.profile['test_prof']
    prof_fk = eq_fk.profile['test_prof']
    
    np.testing.assert_allclose(prof_up.F, prof_fk.F, atol=1e-8)
