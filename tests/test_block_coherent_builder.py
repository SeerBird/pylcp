import numpy as np
import pytest

from pylcp import Hamiltonian, LaserBeams, LaserBeam, MagField, TransitionKey
from pylcp.obe import OBE


def create_two_level_system(include_off_diagonal_mu=False):
    # 2-level system: ground 'g' (N=1), excited 'e' (N=3)
    H_g = np.array([[0.0]])
    H_e = np.diag([1.0, 2.0, 3.0])

    mu_g = np.random.randn(3, 1, 1) + 1j * np.random.randn(3, 1, 1)
    mu_e = np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)
    d_q = np.random.randn(3, 1, 3) + 1j * np.random.randn(3, 1, 3)

    ham = Hamiltonian()
    ham.add_H_0_block('g', H_g)
    ham.add_H_0_block('e', H_e)
    ham.add_mu_q_block('g', 'g', mu_g, muB=1.0)
    ham.add_mu_q_block('e', 'e', mu_e, muB=1.0)

    if include_off_diagonal_mu:
        mu_ge = np.random.randn(3, 1, 3) + 1j * np.random.randn(3, 1, 3)
        ham.add_mu_q_block('g', 'e', mu_ge, muB=1.0)

    ham.add_d_q_block('g', 'e', d_q, gamma=1.5, k=1.0)

    beam = LaserBeam(kvec=np.array([0, 0, 1]), pol=np.array([0, 0, 1]), s=1.0, delta=0.0)
    laser_beams = {TransitionKey('g', 'e'): LaserBeams([beam])}
    mag_field = MagField(lambda r, t: np.array([0.0, 0.0, 1.0]))

    return ham, laser_beams, mag_field


def test_block_coherent_builder_h0():
    ham, laser_beams, mag_field = create_two_level_system()
    obe = OBE(laser_beams, mag_field, ham, transform_into_re_im=False)

    ham.make_full_matrices()
    In = np.eye(ham.n, dtype='complex128')
    baseline_h0 = 1j * (np.kron(ham.H_0.T, In) - np.kron(In, ham.H_0))

    assert np.allclose(obe.ev_mat.h0, baseline_h0, atol=1e-12)


def test_block_coherent_builder_magnetic():
    ham, laser_beams, mag_field = create_two_level_system()
    obe = OBE(laser_beams, mag_field, ham, transform_into_re_im=False)

    ham.make_full_matrices()
    In = np.eye(ham.n, dtype='complex128')

    for q in range(3):
        baseline_mu = 1j * (np.kron(ham.mu_q[q].T, In) - np.kron(In, ham.mu_q[q]))
        assert np.allclose(obe.ev_mat.magnetic[q], baseline_mu, atol=1e-12)


def test_block_coherent_builder_magnetic_off_diagonal():
    ham, laser_beams, mag_field = create_two_level_system(include_off_diagonal_mu=True)
    obe = OBE(laser_beams, mag_field, ham, transform_into_re_im=False)

    In = np.eye(ham.n, dtype='complex128')
    g_slice = ham.get_state_slice('g')
    e_slice = ham.get_state_slice('e')

    fb_g = ham.blocks_by_keys[TransitionKey('g', 'g')]
    fb_e = ham.blocks_by_keys[TransitionKey('e', 'e')]
    fb_ge = ham.blocks_by_keys[TransitionKey('g', 'e')]

    for q in range(3):
        H_mu = np.zeros((ham.n, ham.n), dtype='complex128')
        H_mu[g_slice, g_slice] = fb_g.mu_q.matrix[q] * fb_g.mu_q.muB
        H_mu[e_slice, e_slice] = fb_e.mu_q.matrix[q] * fb_e.mu_q.muB
        H_mu[g_slice, e_slice] = fb_ge.mu_q.matrix[q] * fb_ge.mu_q.muB
        H_mu[e_slice, g_slice] = np.conjugate(fb_ge.mu_q.matrix[q].T) * fb_ge.mu_q.muB

        baseline_mu = 1j * (np.kron(H_mu.T, In) - np.kron(In, H_mu))
        assert np.allclose(obe.ev_mat.magnetic[q], baseline_mu, atol=1e-12)


def test_block_coherent_builder_dq():
    ham, laser_beams, mag_field = create_two_level_system()
    obe = OBE(laser_beams, mag_field, ham, transform_into_re_im=False)

    ham.make_full_matrices()
    In = np.eye(ham.n, dtype='complex128')
    key = TransitionKey('g', 'e')
    gamma = ham.blocks_by_keys[key].d_q.gamma

    for q in range(3):
        M_bare_full = gamma * ham.d_q_bare[key][q] / 4.0
        baseline_dq = 1j * (np.kron(M_bare_full.T, In) - np.kron(In, M_bare_full))
        assert np.allclose(obe.ev_mat.d_q[key][q], baseline_dq, atol=1e-12)

        M_star_full = gamma * ham.d_q_star[key][q] / 4.0
        baseline_dq_star = 1j * (np.kron(M_star_full.T, In) - np.kron(In, M_star_full))
        assert np.allclose(obe.ev_mat.d_q_conj[key][q], baseline_dq_star, atol=1e-12)


def reference_decay_builder(ham):
    ham.make_full_matrices()
    n2 = ham.n ** 2
    ev_mat = np.zeros((n2, n2), dtype='complex128')
    for key in ham.d_q_bare:
        gamma = ham.blocks_by_keys[key].d_q.gamma
        d_q_bare = ham.d_q_bare[key]
        d_q_star = ham.d_q_star[key]
        for ii in range(ham.n):
            for jj in range(ham.n):
                for kk in range(ham.n):
                    for ll in range(ham.n):
                        for mm in range(3):
                            ev_mat[ii + jj * ham.n, ll + jj * ham.n] -= \
                                0.5 * gamma * d_q_star[mm, ll, kk] * d_q_bare[mm, kk, ii]
                            ev_mat[ii + jj * ham.n, kk + ll * ham.n] += \
                                0.5 * gamma * d_q_star[mm, kk, ii] * d_q_bare[mm, jj, ll]
                            ev_mat[ii + jj * ham.n, ll + kk * ham.n] += \
                                0.5 * gamma * d_q_star[mm, ll, ii] * d_q_bare[mm, jj, kk]
                            ev_mat[ii + jj * ham.n, ii + ll * ham.n] -= \
                                0.5 * gamma * d_q_star[mm, jj, kk] * d_q_bare[mm, kk, ll]
    return ev_mat


def test_block_decay_builder():
    ham, laser_beams, mag_field = create_two_level_system()
    obe = OBE(laser_beams, mag_field, ham, transform_into_re_im=False)

    ref_decay = reference_decay_builder(ham)
    assert np.allclose(obe.ev_mat.decay, ref_decay, atol=1e-12)

    # Verify static sum attribute
    assert np.allclose(obe.ev_mat.static, obe.ev_mat.decay + obe.ev_mat.h0, atol=1e-12)

    # Verify recoil velocity and decay rates
    key = TransitionKey('g', 'e')
    assert key in obe.decay_rates
    assert key in obe.recoil_velocity
    assert np.isclose(obe.recoil_velocity[key], 1.0 / ham.mass)


def test_obe_transformed_static():
    # Test with real/imag transformation enabled (default) on a physical system
    from pylcp import hamiltonians, Conventional3DMOTBeams, ConstantMagneticField
    H0_g, muq_g = hamiltonians.singleF(F=0, gF=1.0)
    H0_e, muq_e = hamiltonians.singleF(F=1, gF=1.0)
    dqij = hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    ham = Hamiltonian(H0_g, H0_e, muq_g, muq_e, dqij, mass=87.0)
    beams = Conventional3DMOTBeams(s=1.0, delta=-1.0)
    mag = ConstantMagneticField(np.array([0.0, 0.0, 1.0]))

    obe = OBE(beams, mag, ham, transform_into_re_im=True)

    # Verify self.ev_mat.static is consistent with transformed decay + h0
    assert np.allclose(obe.ev_mat.static, obe.ev_mat.decay + obe.ev_mat.h0, atol=1e-12)
    assert np.all(np.isreal(obe.ev_mat.static))

