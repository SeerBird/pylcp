import numpy as np
import pytest
import tests.vendor.upstream_pylcp as upstream
import pylcp as fork

def create_rateeq_systems():
    """Helper to create matching F_g=0 -> F_e=1 rateeq systems in upstream and fork."""
    # Upstream
    H0_g_up, muq_g_up = upstream.hamiltonians.singleF(F=0, gF=1.0)
    H0_e_up, muq_e_up = upstream.hamiltonians.singleF(F=1, gF=1.0)
    dqij_up = upstream.hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    H_up = upstream.hamiltonian(H0_g_up, H0_e_up, muq_g_up, muq_e_up, dqij_up, mass=87.0)
    beams_up = upstream.conventional3DMOTBeams(s=1.0, delta=-1.0)
    mag_up = upstream.constantMagneticField(np.array([0.0, 0.0, 0.0]))
    eq_up = upstream.rateeq(beams_up, mag_up, H_up)

    # Fork
    H0_g_fk, muq_g_fk = fork.hamiltonians.singleF(F=0, gF=1.0)
    H0_e_fk, muq_e_fk = fork.hamiltonians.singleF(F=1, gF=1.0)
    dqij_fk = fork.hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    H_fk = fork.hamiltonian(H0_g_fk, H0_e_fk, muq_g_fk, muq_e_fk, dqij_fk, mass=87.0)
    beams_fk = fork.conventional3DMOTBeams(s=1.0, delta=-1.0)
    mag_fk = fork.constantMagneticField(np.array([0.0, 0.0, 0.0]))
    eq_fk = fork.rateeq(beams_fk, mag_fk, H_fk)

    return eq_up, eq_fk

def test_rateeq_equilibrium_populations_comparison():
    """Compare rate equation equilibrium populations across positions and velocities."""
    eq_up, eq_fk = create_rateeq_systems()
    
    test_points = [
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([0.1, -0.2, 0.3]), np.array([1.0, 0.0, -0.5])),
    ]
    for r, v in test_points:
        neq_up = eq_up.equilibrium_populations(r, v, t=0)
        neq_fk = eq_fk.equilibrium_populations(r, v, t=0)
        np.testing.assert_allclose(neq_up, neq_fk, atol=1e-10)

def test_rateeq_force_profile_comparison():
    """Compare generated rate equation force profiles."""
    eq_up, eq_fk = create_rateeq_systems()
    
    grid = np.linspace(-1, 1, 5)
    R = np.array([grid, np.zeros(5), np.zeros(5)])
    V = np.array([np.zeros(5), np.zeros(5), np.zeros(5)])
    
    eq_up.generate_force_profile(R, V, name='test_prof')
    eq_fk.generate_force_profile(R, V, name='test_prof')
    
    prof_up = eq_up.profile['test_prof']
    prof_fk = eq_fk.profile['test_prof']
    
    np.testing.assert_allclose(prof_up.F, prof_fk.F, atol=1e-10)
