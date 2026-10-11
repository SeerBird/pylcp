"""
Tools for solving the OBE for laser cooling
author: spe
"""
import copy
import time

import numba
import numpy as np
import scipy.sparse as sparse
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .fields import MagField, LaserBeams
from .hamiltonian import Hamiltonian
from .common import (Progressbar, random_vector, cart2spherical, spherical2cart, BaseForceProfile)
from .governingeq import GoverningEq
from .integration_tools import solve_ivp_random
from .rateeq import RateEq
from .typing import TransitionKey, OBEEvolutionMatrices


@numba.vectorize([numba.float64(numba.complex128), numba.float32(numba.complex64)])
def abs2(x):
    return x.real ** 2 + x.imag ** 2


@numba.jit(nopython=True)
def dot(A, x):
    return A @ x


@numba.jit(nopython=True)
def dot_and_add(A, x, b):
    b += A @ x


def cartesian_vector_tensor_dot(a, B):
    if B.ndim == 2 and a.ndim == 1:
        # Single point:
        return np.dot(B, a)
    elif B.ndim == 2:
        # Constant B, variable a:
        return np.sum(a[np.newaxis, ...] * B[..., np.newaxis], axis=1)
    else:
        # Varaible a and variable B.  Will throw an error if a.shape[1:] != B.shape[2:]:
        return np.sum(a[np.newaxis, ...] * B[...], axis=1)


class ForceProfile(BaseForceProfile):
    """
    Optical Bloch equation force profile

    The force profile object stores all the calculated quantities created by
    the rateeq.generate_force_profile() method.  It has the following
    attributes:

    Attributes
    ----------
    R  : array_like, shape (3, ...)
        Positions at which the force profile was calculated.
    V : array_like, shape (3, ...)
        Velocities at which the force profile was calculated.
    F : array_like, shape (3, ...)
        Total equilibrium force at position R and velocity V.
    f_mag : array_like, shape (3, ...)
        Magnetic force at position R and velocity V.
    f : dictionary of array_like
        The forces due to each laser, indexed by the
        manifold the laser addresses.  The dictionary is keyed by the transition
        driven, and individual lasers are in the same order as in the
        pylcp.LaserBeams object used to create the governing equation.
    f_q : dictionary of array_like
        The force due to each laser and its :math:`q` component, indexed by the
        manifold the laser addresses.  The dictionary is keyed by the transition
        driven, and individual lasers are in the same order as in the
        pylcp.LaserBeams object used to create the governing equation.
    Neq : array_like
        Equilibrium population found.
    """

    def __init__(self, R, V, laserBeams, hamiltonian):
        super().__init__(R, V, laserBeams, hamiltonian)

        self.iterations = np.zeros(self.R[0].shape, dtype='int64')
        self.fq = {}
        for key in laserBeams:
            self.fq[key] = np.zeros(self.R.shape + (3, len(laserBeams[key].beam_vector)))

    def store_data(self, ind, Neq, F, F_laser, F_mag, iterations, F_laser_q):
        super().store_data(ind, Neq, F, F_laser, F_mag)

        for jj in range(3):
            for key in F_laser_q:
                self.fq[key][(jj,) + ind] = F_laser_q[key][jj]

        self.iterations[ind] = iterations


class OBE(GoverningEq):
    """
    The optical Bloch equations

    This class constructs the optical Bloch equations from the given laser
    beams, magnetic field, and hamiltonian.

    Parameters
    ----------
    laserBeams : dictionary of pylcp.LaserBeams, pylcp.LaserBeams
        The laserBeams that will be used in constructing the optical Bloch
        equations.  which transitions in the block diagonal hamiltonian.  It can
        be any of the following:

            * A dictionary of pylcp.LaserBeams: if this is the case, the keys of
              the dictionary should match available :math:`d^{nm}` matrices
              in the pylcp.Hamiltonian object.  The key structure should be
              `n->m`.
            * pylcp.LaserBeams: a single set of laser beams is assumed to
              address the transition `g->e`.

    magField : pylcp.MagField or callable
        The function or object that defines the magnetic field.
    hamiltonian : pylcp.Hamiltonian
        The internal hamiltonian of the particle.
    a : array_like, shape (3,), optional
        A default acceleration to apply to the particle's motion, usually
        gravity. Default: [0., 0., 0.]
    transform_into_re_im : boolean
        Optional flag to transform the optical Bloch equations into real and
        imaginary components.  This helps to decrease computation time as it
        uses the symmetry :math:`\\rho_{ji}=\\rho_{ij}^*` to cut the number
        of equations nearly in half.  Default: True
    use_sparse_matrices : boolean or None
        Optional flag to use sparse matrices.  If none, it will use sparse
        matrices only if the number of internal states > 10, which would result
        in the evolution matrix for the density operators being a 100x100
        matrix.  At that size, there may be some speed up with sparse matrices.
        Default: None
    include_mag_forces : boolean
        Optional flag to include magnetic forces in the force calculation.
        Default: True
    r0 : array_like, shape (3,), optional
        Initial position.  Default: [0., 0., 0.]
    v0 : array_like, shape (3,), optional
        Initial velocity.  Default: [0., 0., 0.]

    Methods
    -------
    """
    hamiltonian:Hamiltonian
    def __init__(self, laserBeams: dict[TransitionKey, LaserBeams] | LaserBeams, magField: MagField,
                 hamiltonian: Hamiltonian,
                 a=np.array([0., 0., 0.]), transform_into_re_im=True,
                 use_sparse_matrices=None, include_mag_forces=True,
                 r0=np.array([0., 0., 0.]), v0=np.array([0., 0., 0.])):
        super().__init__(laserBeams, magField, hamiltonian, a=a, r0=r0, v0=v0)
        # Save the optional arguments:
        self.transform_into_re_im = transform_into_re_im
        self.include_mag_forces = include_mag_forces

        # Should we use sparse matrices?
        if use_sparse_matrices is None:
            if self.hamiltonian.n > 10:  # Generally offers a performance increase
                self.use_sparse_matrices = True
            else:
                self.use_sparse_matrices = False
        else:
            self.use_sparse_matrices = use_sparse_matrices

        # Set up a dictionary to store any resulting force profiles.
        self.profile = {}

        # Reset the current solution to None
        self.sol = None

        # There will be time-dependent and time-independent components of the optical
        # Bloch equations.  The time-independent parts are related to spontaneous
        # emission, applied magnetic field, and the zero-field Hamiltonian.  We
        # compute the latter-two directly from the commutator.

        # Build the matrices that control evolution:
        self.ev_mat = OBEEvolutionMatrices()
        self.__build_ev_matrices()

    # region build evolution matrices
    def __build_ev_matrices(self):
        """Build all evolution matrices and pre-sum the static components."""
        (self.ev_mat.decay, self.decay_rates,
         # the rest is questionable
         self.decay_rates_truncated,
         self.decay_rho_indices,
         self.recoil_velocity) = self._build_decay_ev()
        self.ev_mat.h0 = self.__build_coherent_ev_for_h0()
        self.ev_mat.magnetic = self.__build_coherent_ev_for_mu()
        d_q_ev, d_q_star_ev = self.__build_coherent_ev_for_dq()
        self.ev_mat.d_q = d_q_ev
        self.ev_mat.d_q_conj = d_q_star_ev
        self.ev_mat.static = self.ev_mat.decay + self.ev_mat.h0

        # If necessary, transform the evolution matrices:
        if self.transform_into_re_im:
            self.__transform_ev_matrices()

        if self.use_sparse_matrices:
            self.__convert_to_sparse()

    def __build_coherent_ev_for_h0(self) -> np.ndarray:
        n2 = self.hamiltonian.n ** 2
        h0_ev = np.zeros((n2, n2), dtype='complex128')
        for A in self.hamiltonian.state_labels:
            diag_block = self.hamiltonian.blocks_by_keys[TransitionKey(A, A)]
            if diag_block.h_0 is None:
                continue
            H_AA = diag_block.h_0.matrix
            for C in self.hamiltonian.state_labels:
                c_slice = self.hamiltonian.get_state_slice(C)
                N_C = c_slice.stop - c_slice.start
                In_C = np.eye(N_C, dtype='complex128')

                idx_AC = self.__get_vec_indices_for_labels(A, C)
                h0_ev[np.ix_(idx_AC, idx_AC)] -= 1j * np.kron(In_C, H_AA)

                idx_CA = self.__get_vec_indices_for_labels(C, A)
                h0_ev[np.ix_(idx_CA, idx_CA)] += 1j * np.kron(H_AA.T, In_C)
        return h0_ev

    def __build_coherent_ev_for_mu(self) -> np.ndarray:
        n2 = self.hamiltonian.n ** 2
        mag_ev = np.zeros((3, n2, n2), dtype='complex128')
        for key, fb in self.hamiltonian.blocks_by_keys.items():
            if fb.mu_q is None:
                continue
            A, B = key

            for q in range(3):
                M_AB = fb.mu_q.matrix[q] * fb.mu_q.muB
                if A == B:
                    for C in self.hamiltonian.state_labels:
                        c_slice = self.hamiltonian.get_state_slice(C)
                        N_C = c_slice.stop - c_slice.start
                        In_C = np.eye(N_C, dtype='complex128')

                        idx_AC = self.__get_vec_indices_for_labels(A, C)
                        mag_ev[q][np.ix_(idx_AC, idx_AC)] -= 1j * np.kron(In_C, M_AB)

                        idx_CA = self.__get_vec_indices_for_labels(C, A)
                        mag_ev[q][np.ix_(idx_CA, idx_CA)] += 1j * np.kron(M_AB.T, In_C)
                else:
                    M_BA = np.conjugate(M_AB.T)
                    for C in self.hamiltonian.state_labels:
                        c_slice = self.hamiltonian.get_state_slice(C)
                        N_C = c_slice.stop - c_slice.start
                        In_C = np.eye(N_C, dtype='complex128')

                        # M_AB action
                        idx_AC = self.__get_vec_indices_for_labels(A, C)
                        idx_BC = self.__get_vec_indices_for_labels(B, C)
                        mag_ev[q][np.ix_(idx_AC, idx_BC)] -= 1j * np.kron(In_C, M_AB)

                        idx_CB = self.__get_vec_indices_for_labels(C, B)
                        idx_CA = self.__get_vec_indices_for_labels(C, A)
                        mag_ev[q][np.ix_(idx_CB, idx_CA)] += 1j * np.kron(M_AB.T, In_C)

                        # M_BA action
                        mag_ev[q][np.ix_(idx_BC, idx_AC)] -= 1j * np.kron(In_C, M_BA)
                        mag_ev[q][np.ix_(idx_CA, idx_CB)] += 1j * np.kron(M_BA.T, In_C)
        return mag_ev

    def __build_coherent_ev_for_dq(self) -> tuple[dict, dict]:
        n2 = self.hamiltonian.n ** 2
        d_q_ev = {}
        d_q_star_ev = {}
        for key in self.laserBeams.keys():
            fb = self.hamiltonian.blocks_by_keys[key]
            if fb.d_q is None:
                continue
            gamma = fb.d_q.gamma
            g_label, e_label = key
            d_q_ev[key] = np.zeros((3, n2, n2), dtype='complex128')
            d_q_star_ev[key] = np.zeros((3, n2, n2), dtype='complex128')

            for q in range(3):
                M_ge = (gamma / 4.0) * fb.d_q.matrix[q]
                M_eg = np.conjugate(M_ge.T)

                for C in self.hamiltonian.state_labels:
                    c_slice = self.hamiltonian.get_state_slice(C)
                    N_C = c_slice.stop - c_slice.start
                    In_C = np.eye(N_C, dtype='complex128')

                    idx_gC = self.__get_vec_indices_for_labels(g_label, C)
                    idx_eC = self.__get_vec_indices_for_labels(e_label, C)
                    d_q_ev[key][q][np.ix_(idx_gC, idx_eC)] -= 1j * np.kron(In_C, M_ge)

                    idx_Ce = self.__get_vec_indices_for_labels(C, e_label)
                    idx_Cg = self.__get_vec_indices_for_labels(C, g_label)
                    d_q_ev[key][q][np.ix_(idx_Ce, idx_Cg)] += 1j * np.kron(M_ge.T, In_C)

                    d_q_star_ev[key][q][np.ix_(idx_eC, idx_gC)] -= 1j * np.kron(In_C, M_eg)
                    d_q_star_ev[key][q][np.ix_(idx_Cg, idx_Ce)] += 1j * np.kron(M_eg.T, In_C)

        return d_q_ev, d_q_star_ev

    def _build_decay_ev(self):
        """
        This method constructs the decay portion of the OBE using the radiation
        reaction approximation via block-direct Kronecker assembly.
        """
        decay_rates = {}
        decay_rates_truncated = {}
        decay_rho_indices = {}
        recoil_velocity = {}

        n2 = self.hamiltonian.n ** 2
        decay_total = np.zeros((n2, n2), dtype='complex128')

        for key, fb in self.hamiltonian.blocks_by_keys.items():
            if not fb.d_q is not None:
                continue
            gamma = fb.d_q.gamma
            g_label, e_label = key
            D_q = fb.d_q.matrix  # shape (3, N_g, N_e)

            ev_mat = np.zeros((n2, n2), dtype='complex128')

            # region Loss on excited state, in the vector basis: M_ee = sum over q of I*||d_q||^2
            M_ee = np.zeros((fb.m, fb.m), dtype='complex128')
            for q in range(3):
                M_ee += np.conjugate(D_q[q].T) @ D_q[q]
            # endregion
            # region Loss of coherences/population involving excited state e
            for C in self.hamiltonian.state_labels:
                c_slice = self.hamiltonian.get_state_slice(C)
                N_C = c_slice.stop - c_slice.start
                In_C = np.eye(N_C, dtype='complex128')

                # Fast-index (row) loss on rho_eC: -0.5 * gamma * (In_C (x) M_ee.T)
                idx_eC = self.__get_vec_indices_for_labels(e_label, C)
                ev_mat[np.ix_(idx_eC, idx_eC)] -= 0.5 * gamma * np.kron(In_C, M_ee.T)

                # Slow-index (col) loss on rho_Ce: -0.5 * gamma * (M_ee (x) In_C)
                idx_Ce = self.__get_vec_indices_for_labels(C, e_label)
                ev_mat[np.ix_(idx_Ce, idx_Ce)] -= 0.5 * gamma * np.kron(M_ee, In_C)
            # endregion
            # region Decay from rho_ee to rho_gg: sum over q of gamma * (D_q (x) D_q^*)
            idx_gg = self.__get_vec_indices_for_labels(g_label, g_label)
            idx_ee = self.__get_vec_indices_for_labels(e_label, e_label)
            for q in range(3):
                ev_mat[np.ix_(idx_gg, idx_ee)] += gamma * np.kron(D_q[q], np.conjugate(D_q[q]))
            # endregion
            # Get the decay rates (for the evolve_motion function):
            decay_rates[key] = -np.real(np.array(
                [ev_mat[self.__density_index(ii, ii), self.__density_index(ii, ii)]
                 for ii in range(self.hamiltonian.n)]
            ))
            decay_rates_truncated[key] = decay_rates[key][decay_rates[key] > 0]
            decay_rho_indices[key] = np.array([self.__density_index(ii, ii)
                                               for ii, rate in enumerate(decay_rates[key])
                                               if rate > 0])
            recoil_velocity[key] = fb.d_q.k / self.hamiltonian.mass

            decay_total += ev_mat
        return (decay_total, decay_rates,
                decay_rates_truncated, decay_rho_indices, recoil_velocity)

    # endregion

    def __build_transform_matrices(self):
        self.U = np.zeros((self.hamiltonian.n ** 2, self.hamiltonian.n ** 2),
                          dtype='complex128')
        self.Uinv = np.zeros((self.hamiltonian.n ** 2, self.hamiltonian.n ** 2),
                             dtype='complex128')

        for ii in range(self.hamiltonian.n):
            self.U[self.__density_index(ii, ii),
            self.__density_index(ii, ii)] = 1.
            self.Uinv[self.__density_index(ii, ii),
            self.__density_index(ii, ii)] = 1.

        for ii in range(self.hamiltonian.n):
            for jj in range(ii + 1, self.hamiltonian.n):
                self.U[self.__density_index(ii, jj),
                self.__density_index(ii, jj)] = 1.
                self.U[self.__density_index(ii, jj),
                self.__density_index(jj, ii)] = 1j

                self.U[self.__density_index(jj, ii),
                self.__density_index(ii, jj)] = 1.
                self.U[self.__density_index(jj, ii),
                self.__density_index(jj, ii)] = -1j

        for ii in range(self.hamiltonian.n):
            for jj in range(ii + 1, self.hamiltonian.n):
                self.Uinv[self.__density_index(ii, jj),
                self.__density_index(ii, jj)] = 0.5
                self.Uinv[self.__density_index(ii, jj),
                self.__density_index(jj, ii)] = 0.5

                self.Uinv[self.__density_index(jj, ii),
                self.__density_index(ii, jj)] = -0.5 * 1j
                self.Uinv[self.__density_index(jj, ii),
                self.__density_index(jj, ii)] = +0.5 * 1j

    def __transform_ev_matrix(self, ev_mat):
        if not hasattr(self, 'U'):
            self.__build_transform_matrices()

        ev_mat_new = self.Uinv @ ev_mat @ self.U

        # This should remove the imaginary component.
        if np.allclose(np.imag(ev_mat_new), 0):
            return np.real(ev_mat_new)
        else:
            raise ValueError('Something went dreadfully wrong.')

    def __transform_ev_matrices(self):
        self.ev_mat.decay = self.__transform_ev_matrix(self.ev_mat.decay)
        self.ev_mat.h0 = self.__transform_ev_matrix(self.ev_mat.h0)
        self.ev_mat.static = self.__transform_ev_matrix(self.ev_mat.static)

        self.ev_mat['reE'] = {}
        self.ev_mat['imE'] = {}
        for key in self.ev_mat['d_q'].keys():
            self.ev_mat['reE'][key] = np.array([self.__transform_ev_matrix(
                self.ev_mat['d_q'][key][jj] + self.ev_mat['d_q*'][key][jj]
            ) for jj in range(3)])
            # Unclear why the following works, I calculate that there should
            # be a minus sign out front.
            self.ev_mat['imE'][key] = np.array([self.__transform_ev_matrix(
                1j * (self.ev_mat['d_q'][key][jj] - self.ev_mat['d_q*'][key][jj])
            ) for jj in range(3)])

        # Transform Bq back into Bx, By, and Bz (making it real):
        self.ev_mat['B'] = spherical2cart(self.ev_mat['B'])

        for jj in range(3):
            self.ev_mat['B'][jj] = self.__transform_ev_matrix(self.ev_mat['B'][jj])
        self.ev_mat['B'] = np.real(self.ev_mat['B'])

        del self.ev_mat['d_q']
        del self.ev_mat['d_q*']

    def __convert_to_sparse(self):
        def convert_based_on_shape(matrix):
            # Vector:
            if matrix.shape == (3, self.hamiltonian.n ** 2, self.hamiltonian.n ** 2):
                new_list = [None] * 3
                for jj in range(3):
                    new_list[jj] = sparse.csr_matrix(matrix[jj])

                return new_list
            # Scalar:
            else:
                return sparse.csr_matrix(matrix)

        for key in self.ev_mat:
            if isinstance(self.ev_mat[key], dict):
                for subkey in self.ev_mat[key]:
                    self.ev_mat[key][subkey] = convert_based_on_shape(
                        self.ev_mat[key][subkey]
                    )
            else:
                self.ev_mat[key] = convert_based_on_shape(self.ev_mat[key])

    def __reshape_rho(self, rho):
        if self.transform_into_re_im:
            rho = rho.astype('complex128')

            if len(rho.shape) == 1:
                rho = self.U @ rho
            else:
                for jj in range(rho.shape[1]):
                    rho[:, jj] = self.U @ rho[:, jj]

        rho = rho.reshape((self.hamiltonian.n, self.hamiltonian.n) +
                          rho.shape[1:])

        """# If not:
        if self.transform_into_re_im:
            new_rho = np.zeros(rho.shape, dtype='complex128')
            for jj in range(new_rho.shape[2]):
                new_rho[:, :, jj] = (np.diag(np.diagonal(rho[:, :, jj])) +
                                     np.triu(rho[:, :, jj], k=1) +
                                     np.triu(rho[:, :, jj], k=1).T +
                                     1j*np.tril(rho[:, :, jj], k=-1) -
                                     1j*np.tril(rho[:, :, jj], k=-1).T)
            rho = new_rho"""

        return rho

    def __reshape_sol(self):
        """
        Reshape the solution to have all the proper parts.
        """
        self.sol.rho = self.__reshape_rho(self.sol.y[:-6])
        self.sol.r = np.real(self.sol.y[-3:])
        self.sol.v = np.real(self.sol.y[-6:-3])

        del self.sol.y

    def set_initial_rho(self, rho0):
        """
        Sets the initial :math:`\\rho` matrix

        Parameters
        ----------
        rho0 : array_like
            The initial :math:`\\rho`.  It must have :math:`n^2` elements, where :math:`n`
            is the total number of states in the system.  If a flat array, it
            will be reshaped.
        """
        if np.any(np.isnan(rho0)) or np.any(np.isinf(rho0)):
            raise ValueError('rho0 has NaNs or Infs!')

        if rho0.size != self.hamiltonian.n ** 2:
            raise ValueError('rho0 should have n^2 elements.')

        if rho0.shape == (self.hamiltonian.n, self.hamiltonian.n):
            rho0 = rho0.flatten()

        if self.transform_into_re_im and rho0.dtype is np.dtype('complex128'):
            self.rho0 = self.Uinv @ rho0
        elif (not self.transform_into_re_im and
              not rho0.dtype is np.dtype('complex128')):
            self.rho0 = rho0.astype('complex128')
        else:
            self.rho0 = rho0

    def set_initial_rho_equally(self):
        """
        Sets the initial :math:`\\rho` matrix such that all states have the same
        population.
        """
        if self.transform_into_re_im:
            self.rho0 = np.zeros((self.hamiltonian.n ** 2,))
        else:
            self.rho0 = np.zeros((self.hamiltonian.n ** 2,), dtype='complex128')

        for jj in range(self.hamiltonian.ns[0]):
            self.rho0[self.__density_index(jj, jj)] = 1 / self.hamiltonian.ns[0]

    def set_initial_rho_from_populations(self, Npop):
        """
        Sets the diagonal elements of the initial :math:`\\rho` matrix

        Parameters
        ----------
        Npop : array_like
            Array of the initial populations of the states in the system.  The
            length must be :math:`n`, where :math:`n` is the number of states.
        """
        if self.transform_into_re_im:
            self.rho0 = np.zeros((self.hamiltonian.n ** 2,))
        else:
            self.rho0 = np.zeros((self.hamiltonian.n ** 2,), dtype='complex128')

        if len(Npop) != self.hamiltonian.n:
            raise ValueError('Npop has only %d entries for %d states.' %
                             (len(Npop), self.hamiltonian.n))
        if np.any(np.isnan(Npop)) or np.any(np.isinf(Npop)):
            raise ValueError('Npop has NaNs or Infs!')

        Npop = Npop / np.sum(Npop)  # Just make sure it is normalized.
        for jj in range(self.hamiltonian.n):
            self.rho0[self.__density_index(jj, jj)] = Npop[jj]

    def set_initial_rho_from_rateeq(self):
        """
        Sets the diagonal elements of the initial :math:`\\rho` matrix using
        the equilibrium populations as determined by pylcp.rateeq
        """
        if not hasattr(self, 'rateeq'):
            self.rateeq = RateEq(self.laserBeams, self.magField, self.hamiltonian)
        Neq = self.rateeq.equilibrium_populations(self.r0, self.v0, t=0)
        self.set_initial_rho_from_populations(Neq)

    def full_OBE_ev_scratch(self, r, t):
        """
        Calculate the evolution for the density matrix

        This function calculates the OBE evolution matrix at position t and r
        from scratch, first computing the full Hamiltonian, then the
        OBE evolution matrix computed via commutators, then adding in the decay
        matrix evolution.  If `Bq` is `None`, it will compute `Bq`.

        Parameters
        ----------
        r : array_like, shape (3,)
            Position at which to calculate evolution matrix
        t : float
            Time at which to calculate evolution matrix

        Returns
        -------
        ev_mat : array_like
            Evolution matrix for the densities
        """
        Eq = {}
        for key in self.laserBeams.keys():
            Eq[key] = self.laserBeams[key].total_electric_field(r, t)

        B = self.magField.Field(r, t)
        Bq = cart2spherical(B)

        H = self.hamiltonian.return_full_H(Bq, Eq)
        ev_mat = self.__build_coherent_ev_submatrix(H)

        if self.transform_into_re_im:
            return self.__transform_ev_matrix(ev_mat + self.ev_mat['decay'])
        else:
            return ev_mat + self.ev_mat['decay']

    def full_OBE_ev(self, r, t):
        """
        Calculate the evolution for the density matrix

        This function calculates the OBE evolution matrix by assembling
        pre-stored versions of the component matries.  This should be
        significantly faster than full_OBE_ev_scratch, but it may suffer bugs
        in the evolution that full_OBE_ev_scratch will not. If Bq is None,
        it will compute Bq based on r, t

        Parameters
        ----------
        r : array_like, shape (3,)
            Position at which to calculate evolution matrix
        t : float
            Time at which to calculate evolution matrix

        Returns
        -------
        ev_mat : array_like
            Evolution matrix for the densities
        """
        ev_mat = self.ev_mat.static.copy()

        # Add in electric fields:
        for key in self.laserBeams.keys():
            if self.transform_into_re_im:
                Eq = self.laserBeams[key].total_electric_field(r, t)
                for ii in range(3):
                    if np.abs(Eq[ii]) > 1e-10:
                        ev_mat -= np.real(Eq[ii]) * self.ev_mat['reE'][key][ii]
                        ev_mat -= np.imag(Eq[ii]) * self.ev_mat['imE'][key][ii]
            else:
                Eq = self.laserBeams[key].total_electric_field(np.real(r), t)
                for ii in range(3):
                    if np.abs(Eq[ii]) > 1e-10:
                        ev_mat -= np.conjugate(Eq[ii]) * self.ev_mat['d_q'][key][ii]
                        ev_mat -= Eq[ii] * self.ev_mat['d_q*'][key][ii]

        # Add in magnetic fields:
        B = self.magField.Field(r, t)
        for ii, q in enumerate(range(-1, 2)):
            if self.transform_into_re_im:
                if np.abs(Bq[ii]) > 1e-10:
                    drhodt -= self.ev_mat['B'][ii] * B[ii] @ rho
            else:
                Bq = cart2spherical(B)
                if np.abs(Bq[2 - ii]) > 1e-10:
                    drhodt -= (-1) ** np.abs(q) * self.ev_mat['B'][ii] * Bq[2 - ii] @ rho

        return ev_mat

    def __drhodt(self, r, t, rho):
        """
        It is MUCH more efficient to do matrix vector products and add the
        results together rather than to add the matrices together (as above)
        and then do the dot.  It is also most efficient to avoid doing useless
        math if the applied field is zero.
        """
        drhodt = self.ev_mat.static @ rho

        # Add in electric fields:
        for key in self.laserBeams.keys():
            if self.transform_into_re_im:
                Eq = self.laserBeams[key].total_electric_field(r, t)
                for ii, q in enumerate([-1.0, 0.0, 1.0]):
                    if np.abs(Eq[2 - ii]) > 1e-10:
                        drhodt -= ((-1.) ** q * np.real(Eq[2 - ii]) *
                                   (self.ev_mat['reE'][key][ii] @ rho))
                        drhodt -= ((-1.) ** q * np.imag(Eq[2 - ii]) *
                                   (self.ev_mat['imE'][key][ii] @ rho))
            else:
                Eq = self.laserBeams[key].total_electric_field(np.real(r), t)
                for ii, q in enumerate([-1.0, 0.0, 1.0]):
                    if np.abs(Eq[2 - ii]) > 1e-10:
                        drhodt -= ((-1.) ** q * Eq[2 - ii] *
                                   (self.ev_mat['d_q'][key][ii] @ rho))
                        drhodt -= ((-1.) ** q * np.conjugate(Eq[2 - ii]) *
                                   (self.ev_mat['d_q*'][key][ii] @ rho))

        # Add in magnetic fields:
        B = self.magField.Field(r, t)
        for ii, q in enumerate(range(-1, 2)):
            if self.transform_into_re_im:
                if np.abs(B[ii]) > 1e-10:
                    drhodt -= self.ev_mat['B'][ii] * B[ii] @ rho
            else:
                Bq = cart2spherical(B)
                if np.abs(Bq[ii]) > 1e-10:
                    drhodt -= self.ev_mat['B'][ii] * np.conjugate(Bq[ii]) @ rho

        return drhodt

    def evolve_density(self, t_span, progress_bar=False, **kwargs):
        """
        Evolve the density operators :math:`\\rho_{ij}` in time.

        This function integrates the optical Bloch equations to determine how
        the populations evolve in time.  Any initial velocity is kept constant
        while the atom potentially moves through the light field.  This function
        is therefore useful in determining average forces.  Any constant
        acceleration set when the OBEs were generated is ignored. It is
        analogous to rateeq.evolve_populations().

        Parameters
        ----------
        t_span : list or array_like
            A two element list or array that specify the initial and final time
            of integration.
        progress_bar : boolean
            Show a progress bar as the calculation proceeds.  Default:False
        **kwargs :
            Additional keyword arguments get passed to solve_ivp, which is
            what actually does the integration.

        Returns
        -------
        sol : OdeSolution
            Bunch object that contains the following fields:

                * t: integration times found by solve_ivp
                * rho: density matrix
                * v: atomic velocity (constant)
                * r: atomic position

            It contains other important elements, which can be discerned from
            scipy's solve_ivp documentation.
        """
        a = np.zeros((3,))

        if progress_bar:
            progress = Progressbar()

        def dydt(t, y):
            if progress_bar and t <= t_span[1]:
                progress.update(t / t_span[1])

            return np.concatenate((self.__drhodt(y[-3:], t, y[:-6]), a, y[-6:-3]))

        self.sol = solve_ivp(dydt, t_span,
                             np.concatenate((self.rho0, self.v0, self.r0)),
                             **kwargs)

        if progress_bar:
            # Just in case the solve_ivp_random terminated due to an event.
            progress.update(1.)

        # Remake the solution:
        self.__reshape_sol()

        return self.sol

    def evolve_motion(self, t_span, freeze_axis=[False, False, False],
                      random_recoil=False, max_scatter_probability=0.1,
                      progress_bar=False, record_force=False,
                      rng=np.random.default_rng(), **kwargs):
        # TODO: clean this mess up, avoid storing as much in OBE etc.
        """
        Evolve :math:`\\rho_{ij}` and the motion of the atom in time.

        This function evolves the optical Bloch equations, moving the atom
        along given the instantaneous force, for some period of time.

        Parameters
        ----------
        t_span : list or array_like
            A two element list or array that specify the initial and final time
            of integration.
        freeze_axis : list of boolean
            Freeze atomic motion along the specified axis.
            Default: [False, False, False]
        random_recoil : boolean
            Allow the atom to randomly recoil from scattering events.
            Default: False
        max_scatter_probability : float
            When undergoing random recoils, this sets the maximum time step such
            that the maximum scattering probability is less than or equal to
            this number during the next time step.  Default: 0.1
        progress_bar : boolean
            If true, show a progress bar as the calculation proceeds.
            Default: False
        record_force : boolean
            If true, record the instantaneous force and store in the solution.
            Default: False
        rng : numpy.random.Generator()
            A properly-seeded random number generator.  Default: calls
            ``numpy.random.default.rng()``
        **kwargs :
            Additional keyword arguments get passed to solve_ivp_random, which
            is what actually does the integration.

        Returns
        -------
        sol : OdeSolution
            Bunch object that contains the following fields:

                * t: integration times found by solve_ivp
                * rho: density matrix
                * v: atomic velocity
                * r: atomic position

            It contains other important elements, which can be discerned from
            scipy's solve_ivp documentation.
        """
        free_axes = np.bitwise_not(freeze_axis)

        if progress_bar:
            progress = Progressbar()
        else:
            progress = None

        if record_force:
            ts = []
            Fs = []

        def dydt(t, y):
            if progress_bar:
                progress.update(t / t_span[1])

            if record_force:
                F = self.force(y[-3:], t, y[:-6], return_details=True)

                ts.append(t)
                Fs.append(F)

                F = F[0]
            else:
                F = self.force(y[-3:], t, y[:-6], return_details=False)

            return np.concatenate((
                self.__drhodt(y[-3:], t, y[:-6]),
                F * free_axes / self.hamiltonian.mass + self.constant_accel,
                y[-6:-3]
            ))

        def random_recoil_function(t, y, dt):
            num_of_scatters = 0
            total_P = 0.

            # Go over each block in the Hamiltonian and compute the decay:
            for key in self.decay_rates:
                P = dt * self.decay_rates_truncated[key] * np.real(y[self.decay_rho_indices[key]])

                # Roll the dice N times, where $N=\sum(n_i)
                dice = rng.random(len(P))

                # For any random number that is lower than P_i, add a
                # recoil velocity.
                for ii in range(np.sum(dice < P)):
                    num_of_scatters += 1
                    y[-6:-3] += self.recoil_velocity[key] * (random_vector(rng, free_axes) +
                                                             random_vector(rng, free_axes))

                # Save the total probability of a scatter:
                total_P += np.sum(P)

            # Calculate a new maximum dt to make sure we evolve while not
            # exceeding dt max:
            new_dt_max = (max_scatter_probability / total_P) * dt

            return num_of_scatters, new_dt_max

        if not random_recoil:
            self.sol = solve_ivp(
                dydt, t_span, np.concatenate((self.rho0, self.v0, self.r0)),
                **kwargs)
        else:
            self.sol = solve_ivp_random(
                dydt, random_recoil_function, t_span,
                np.concatenate((self.rho0, self.v0, self.r0)),
                **kwargs
            )

        if progress is not None:
            # Just in case the solve_ivp_random terminated due to an event.
            progress.update(1.)

        # Remake the solution:
        self.__reshape_sol()

        # Interpolate the force:
        if record_force:
            f = interp1d(ts[:-1], np.array([f[0] for f in Fs[:-1]]).T)
            self.sol.F = f(self.sol.t)

            f = interp1d(ts[:-1], np.array([f[3] for f in Fs[:-1]]).T)
            self.sol.fmag = f(self.sol.t)

            self.sol.f = {}
            for key in Fs[0][1]:
                f = interp1d(ts[:-1], np.array([f[1][key] for f in Fs[:-1]]).T)
                self.sol.f[key] = f(self.sol.t)
                self.sol.f[key] = np.swapaxes(self.sol.f[key], 0, 1)

        return self.sol

    def observable(self, O, rho=None):
        """
        Observable returns the obervable O given density matrix rho.

        Parameters
        ----------
        O : array or array-like
            The matrix form of the observable operator.  Can have any shape,
            representing scalar, vector, or tensor operators, but the last two
            axes must correspond to the matrix of the operator and have the
            same dimensions of the generating Hamiltonian.  For example,
            a vector operator might have the shape (3, n, n), where n
            is the number of states and the first axis corresponds to x, y,
            and z.
        rho : [optional] array or array-like
            The density matrix.  The first two dimensions must have sizes
            (n, n), but there may be multiple instances of the density matrix
            tiled in the higher dimensions.  For example, a rho with (n, n, m)
            could have m instances of the density matrix at different times.

            If not specified, will get rho from the current solution stored in
            memory.

        Returns
        -------
        observable : float or array
            observable has shape (O[:-2])+(rho[2:])
        """
        if rho is None:
            rho = self.sol.rho

        if rho.shape[:2] != (self.hamiltonian.n, self.hamiltonian.n):
            raise ValueError('rho must have dimensions (n, n,...), where n ' +
                             'corresponds to the number of states in the ' +
                             'generating Hamiltonian. ' +
                             'Instead, shape of rho is %s.' % str(rho.shape))
        elif O.shape[-2:] != (self.hamiltonian.n, self.hamiltonian.n):
            raise ValueError('O must have dimensions (..., n, n), where n ' +
                             'corresponds to the number of states in the ' +
                             'generating Hamiltonian. ' +
                             'Instead, shape of O is %s.' % str(O.shape))
        else:
            avO = np.tensordot(O, rho, axes=[(-2, -1), (0, 1)])
            if np.allclose(np.imag(avO), 0):
                return np.real(avO)
            else:
                return avO

    def force(self, r, t, rho, return_details=False):
        """
        Calculates the instantaneous force

        Parameters
        ----------
        r : array_like
            Position at which to calculate the force
        t : float
            Time at which to calculate the force
        rho : array_like
            Density matrix with which to calculate the force
        return_details : boolean, optional
            If true, returns the forces from each laser and the scattering rate
            matrix.

        Returns
        -------
        F : array_like
            total equilibrium force experienced by the atom
        F_laser : array_like
            If return_details is True, the forces due to each laser.
        F_laser_q : array_like
            If return_details is True, the forces due to each laser and it's
            q component of the polarization.
        F_mag : array_like
            If return_details is True, the forces due to the magnetic field.
        """
        if rho.shape[0] != self.hamiltonian.n:
            rho = self.__reshape_rho(rho)

        f = np.zeros((3,) + rho.shape[2:])
        if return_details:
            f_laser_q = {}
            f_laser = {}

        for key in self.laserBeams:
            # First, determine the average mu_q:
            # This returns a (3,) + rho.shape[2:] array
            mu_q_av = self.observable(self.hamiltonian.d_q_bare[key], rho)
            gamma = self.hamiltonian.blocks_by_keys[key].d_q.gamma

            if not return_details:
                delE = self.laserBeams[key].total_electric_field_gradient(np.real(r), t)

                # We are just looking at the d_q, whereas the full observable
                # is \nabla (d_q \cdot E^\dagger) + (d_q^* E)) =
                # 2 Re[\nabla (d_q\cdot E^\dagger)].  Putting in the units,
                # we see we need a factor of gamma/4, making
                # this 2 Re[\nabla (d_q\cdot E^\dagger)]/4 =
                # Re[\nabla (d_q\cdot E^\dagger)]/2
                for jj, q in enumerate(np.arange(-1., 2., 1.)):
                    f += np.real((-1) ** q * gamma * mu_q_av[jj] * delE[:, 2 - jj]) / 2
            else:
                f_laser_q[key] = np.zeros((3, 3, self.laserBeams[key].num_of_beams)
                                          + rho.shape[2:])
                f_laser[key] = np.zeros((3, self.laserBeams[key].num_of_beams)
                                        + rho.shape[2:])

                # Now, dot it into each laser beam:
                for ii, beam in enumerate(self.laserBeams[key].beam_vector):
                    if not self.transform_into_re_im:
                        delE = beam.electric_field_gradient(np.real(r), t)
                    else:
                        delE = beam.electric_field_gradient(r, t)

                    for jj, q in enumerate(np.arange(-1., 2., 1.)):
                        f_laser_q[key][:, jj, ii] += \
                            np.real((-1) ** q * gamma * mu_q_av[jj] * delE[:, 2 - jj]) / 2

                    f_laser[key][:, ii] = np.sum(f_laser_q[key][:, :, ii], axis=1)

                f += np.sum(f_laser[key], axis=1)

        # Are we including magnetic forces?
        if self.include_mag_forces:
            # This function returns a matrix that is either (3, 3) (if constant)
            # or (3, 3, t.size).  The first two dimensions are like
            # [dBx/dx, dBy/dx, dBz/dx; dBx/dy, dBy/dy, dBz/dy], and so on.
            # We need to dot, and su
            delB = self.magField.gradient(np.real(r))

            # What's the expectation value of mu?  Returns (3,) or (3, t.size)
            av_mu = self.observable(self.hamiltonian.mu, rho)

            # Now dot it into the gradient:
            f_mag = cartesian_vector_tensor_dot(av_mu, delB)

            # Add it into the regular force.
            f += f_mag
        elif return_details:
            f_mag = np.zeros(f.shape)

        if return_details:
            return f, f_laser, f_laser_q, f_mag
        else:
            return f

    def find_equilibrium_force(self, deltat=500, itermax=100, Npts=5001,
                               rel=1e-5, abs=1e-9, debug=False,
                               initial_rho='rateeq',
                               return_details=False, **kwargs):
        """
        Finds the equilibrium force at the initial position

        This method works by solving the OBEs in a chunk of time
        :math:`\\Delta T`, calculating the force during that chunck, continuing
        the integration for another chunck, calculating the force during that
        subsequent chunck, and comparing the average of the forces of the two
        chunks to see if they have converged.

        Parameters
        ----------
        deltat : float
            Chunk time :math:`\\Delta T`.  Default: 500
        itermax : int, optional
            Maximum number of iterations.  Default: 100
        Npts : int, optional
            Number of points to divide the chunk into.  Default: 5001
        rel : float, optional
            Relative convergence parameter.  Default: 1e-5
        abs : float, optional
            Absolute convergence parameter.  Default: 1e-9
        debug : boolean, optional
            If true, pring out debug information as it goes.
        initial_rho : 'rateeq' or 'equally'
            Determines how to set the initial rho at the start of the
            calculation.
        return_details : boolean, optional
            If true, returns the forces from each laser and the scattering rate
            matrix.

        Returns
        -------
        F : array_like
            total equilibrium force experienced by the atom
        F_laser : dictionary of array_like
            If return_details is True, the forces due to each laser, indexed
            by the manifold the laser addresses.  The dictionary is keyed by
            the transition driven, and individual lasers are in the same order
            as in the pylcp.laserBeams object used to create the governing
            equation.
        F_laser : dictionary of array_like
            If return_details is True, the forces due to each laser and its q
            component, indexed by the manifold the laser addresses.  The
            dictionary is keyed by the transition driven, and individual lasers
            are in the same order as in the pylcp.laserBeams object used to
            create the governing equation.
        F_mag : array_like
            If return_details is True, the forces due to the magnetic field.
        Neq : array_like
            If return_details is True, the equilibrium populations.
        ii : int
            Number of iterations needed to converge.
        """
        if initial_rho == 'rateeq':
            self.set_initial_rho_from_rateeq()
        elif initial_rho == 'equally':
            self.set_initial_rho_equally()
        elif initial_rho == 'frompops':
            Npop = kwargs.pop('init_pop', None)
            self.set_initial_rho_from_populations(Npop)
        else:
            raise ValueError('Argument initial_rho=%s not understood' % initial_rho)

        old_f_avg = np.array([np.inf, np.inf, np.inf])

        if debug:
            print('Finding equilbrium force at ' +
                  'r=(%.2f, %.2f, %.2f) ' % (self.r0[0], self.r0[1], self.r0[2]) +
                  'v=(%.2f, %.2f, %.2f) ' % (self.v0[0], self.v0[1], self.v0[2]) +
                  'with deltat = %.2f, itermax = %d, Npts = %d, ' % (deltat, itermax, Npts) +
                  'rel = %.1e and abs = %.1e' % (rel, abs)
                  )
            self.piecewise_sols = []

        ii = 0
        while True:
            if not Npts is None:
                kwargs['t_eval'] = np.linspace(ii * deltat, (ii + 1) * deltat, int(Npts))

            self.evolve_density([ii * deltat, (ii + 1) * deltat], **kwargs)
            f, f_laser, f_laser_q, f_mag = self.force(self.sol.r, self.sol.t, self.sol.rho,
                                                      return_details=True)

            f_avg = np.mean(f, axis=1)

            if debug:
                print(ii, f_avg, np.sum(f_avg ** 2))
                self.piecewise_sols.append(self.sol)

            if (np.sum((f_avg) ** 2) < abs or
                    np.sum((old_f_avg - f_avg) ** 2) / np.sum((f_avg) ** 2) < rel or
                    np.sum((old_f_avg - f_avg) ** 2) < abs):
                break;
            elif ii >= itermax - 1:
                break;
            else:
                old_f_avg = copy.copy(f_avg)
                self.set_initial_rho(self.sol.rho[:, :, -1])
                self.set_initial_position_and_velocity(self.sol.r[:, -1],
                                                       self.sol.v[:, -1])
                ii += 1

        if return_details:
            f_mag = np.mean(f_mag, axis=1)

            f_laser_avg = {}
            f_laser_avg_q = {}
            for key in f_laser:
                f_laser_avg[key] = np.mean(f_laser[key], axis=2)
                f_laser_avg_q[key] = np.mean(f_laser_q[key], axis=3)

            Neq = np.real(np.diagonal(np.mean(self.sol.rho, axis=2)))
            return (f_avg, f_laser_avg, f_laser_avg_q, f_mag, Neq, ii)
        else:
            return f_avg

    def generate_force_profile(self, R, V, name=None, progress_bar=False,
                               **kwargs):
        """
        Map out the equilibrium force vs. position and velocity

        Parameters
        ----------
        R : array_like, shape(3, ...)
            Position vector.  First dimension of the array must be length 3, and
            corresponds to :math:`x`, :math:`y`, and :math:`z` components,
            repsectively.
        V : array_like, shape(3, ...)
            Velocity vector.  First dimension of the array must be length 3, and
            corresponds to :math:`v_x`, :math:`v_y`, and :math:`v_z` components,
            repsectively.
        name : str, optional
            Name for the profile.  Stored in profile dictionary in this object.
            If None, uses the next integer, cast as a string, (i.e., '0') as
            the name.
        progress_bar : boolean, optional
            Displays a progress bar as the proceeds.  Default: False

        Returns
        -------
        profile : pylcp.obe.ForceProfile
            Resulting force profile.
        """

        def default_deltat(r, v, deltat_v, deltat_r, deltat_tmax):
            deltat = None
            if deltat_v is not None:
                vabs = np.sqrt(np.sum(v ** 2))
                if vabs == 0.:
                    deltat = deltat_tmax
                else:
                    deltat = np.min([2 * np.pi * deltat_v / vabs, deltat_tmax])

            if deltat_r is not None:
                rabs = np.sqrt(np.sum(r ** 2))
                if rabs == 0.:
                    deltat = deltat_tmax
                else:
                    deltat = np.min([2 * np.pi * deltat_r / rabs, deltat_tmax])

            return deltat

        deltat_r = kwargs.pop('deltat_r', None)
        deltat_v = kwargs.pop('deltat_v', None)
        deltat_tmax = kwargs.pop('deltat_tmax', np.inf)
        deltat_func = kwargs.pop(
            'deltat_func',
            lambda r, v: default_deltat(r, v, deltat_v, deltat_r, deltat_tmax)
        )

        if not name:
            name = '{0:d}'.format(len(self.profile))

        self.profile[name] = ForceProfile(R, V, self.laserBeams, self.hamiltonian)

        it = np.nditer([R[0], R[1], R[2], V[0], V[1], V[2]],
                       flags=['refs_ok', 'multi_index'],
                       op_flags=[['readonly'], ['readonly'], ['readonly'],
                                 ['readonly'], ['readonly'], ['readonly']])

        if progress_bar:
            progress = Progressbar()

        for (x, y, z, vx, vy, vz) in it:
            # Construct the rate equations:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])

            if progress_bar:
                tic = time.time()

            self.set_initial_position_and_velocity(r, v)

            if not deltat_func(r, v) is None:
                kwargs['deltat'] = deltat_func(r, v)
            kwargs['return_details'] = True

            F, F_laser, F_laser_q, F_mag, Neq, iterations = self.find_equilibrium_force(**kwargs)

            self.profile[name].store_data(it.multi_index, Neq, F, F_laser, F_mag,
                                          iterations, F_laser_q)

            if progress_bar:
                progress.update((it.iterindex + 1) / it.itersize)

        return self.profile[name]

    # region index and matrix helpers
    def __density_index(self, ii, jj):
        """
        This function returns the index in the rho vector that corresponds to element rho_{ij}.  If
        """
        return ii + jj * self.hamiltonian.n

    def __build_coherent_ev_submatrix(self, H):
        """
        This method builds the coherent evolution based on a submatrix of the
        Hamiltonian H using vectorized Kronecker products.
        """
        n = self.hamiltonian.n
        if getattr(self, 'use_sparse_matrices', False):
            In = sparse.eye(n, dtype='complex128', format='csr')
            H_sp = sparse.csr_matrix(H)
            return 1j * (
                    sparse.kron(H_sp.T, In, format='csr') - sparse.kron(In, H_sp, format='csr'))
        else:
            In = np.eye(n, dtype='complex128')
            return 1j * (np.kron(H.T, In) - np.kron(In, H))

    def __get_vec_indices_for_labels(self, row_label: str, col_label: str) -> np.ndarray:
        r_slice = self.hamiltonian.get_state_slice(row_label)
        c_slice = self.hamiltonian.get_state_slice(col_label)
        r_idx = np.arange(r_slice.start, r_slice.stop)
        c_idx = np.arange(c_slice.start, c_slice.stop)
        ii, jj = np.meshgrid(r_idx, c_idx, indexing='ij')
        return (ii + jj * self.hamiltonian.n).ravel(order='F')

    # endregion
