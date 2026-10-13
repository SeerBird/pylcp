from __future__ import annotations
import time
import copy
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from pylcp.hamiltonians import wig6j, wig3j
from pylcp.typing import Polarization, PolarizationEnum

if TYPE_CHECKING:
    from pylcp import Hamiltonian


# region maths
def cart2spherical(A):
    return np.array([(A[0] - 1j * A[1]) / np.sqrt(2), A[2], -(A[0] + 1j * A[1]) / np.sqrt(2)])


def spherical2cart(A):
    return np.array([1 / np.sqrt(2) * (-A[2] + A[0]), 1j / np.sqrt(2) * (A[2] + A[0]), A[1]])


def spherical_dot(A, B):
    return np.tensordot(A, np.array([-1., 1., -1.]) * B[::-1], axes=(0, 0))


def project_polarization_spherical(kvec: np.ndarray, pol: Polarization) -> np.ndarray:
    """Project a laser beam's polarization (PolarizationEnum or vector)
    onto the atomic quantization axis z using pure numpy Wigner D-matrix rotation.
    """
    if isinstance(pol, np.ndarray) and pol.shape == (3,):
        return pol

    k_norm = np.linalg.norm(kvec)
    k_hat = np.array([0.0, 0.0, 1.0]) if k_norm == 0 else np.array(kvec, dtype=float) / k_norm

    cosbeta = k_hat[2]
    sinbeta = np.sqrt(max(0.0, 1.0 - cosbeta ** 2))
    gamma = np.arctan2(k_hat[1], k_hat[0]) if abs(cosbeta) < 1.0 else 0.0

    D = np.array([
        [(1 + cosbeta) / 2 * np.exp(1j * gamma), -sinbeta / np.sqrt(2),
         (1 - cosbeta) / 2 * np.exp(-1j * gamma)],
        [sinbeta / np.sqrt(2) * np.exp(1j * gamma), cosbeta,
         -sinbeta / np.sqrt(2) * np.exp(-1j * gamma)],
        [(1 - cosbeta) / 2 * np.exp(1j * gamma), sinbeta / np.sqrt(2),
         (1 + cosbeta) / 2 * np.exp(-1j * gamma)]
    ])
    pol: int | float | PolarizationEnum
    # Convert enum or scalar to beam frame spherical basis
    if pol == PolarizationEnum.SIGMA_PLUS or pol in (+1, 1.0):
        p_beam = np.array([0.0, 0.0, 1.0], dtype=complex)
    elif pol == PolarizationEnum.SIGMA_MINUS or pol in (-1, -1.0):
        p_beam = np.array([1.0, 0.0, 0.0], dtype=complex)
    else:  # PI / 0
        p_beam = np.array([0.0, 1.0, 0.0], dtype=complex)

    return np.linalg.inv(D) @ p_beam


def random_vector(rng, free_axes=None):
    """
    This function returns a random vector in either 1D, 2D or 3D

    Parameters:
    -----------
        rng: numpy.random.Generator()
            Properly seeded random number generator.
        free_axis: list of 3 boolean (option)
            Which axes (x, y, z) are considered free axes.  Default:
            [True, True, True], i.e., a 3D vector.

    Returns
    -------
        vector: array_like of shape (3,)
            Random vector with unit length.
    """
    if free_axes is None:
        free_axes = [True, True, True]
    if np.sum(free_axes) == 1:
        return (np.sign(rng.random(1) - 0.5) * free_axes).astype('float64')
    elif np.sum(free_axes) == 2:
        phi = 2 * np.pi * rng.random(1)[0]
        x = np.array([np.cos(phi), np.sin(phi)])
        y = np.zeros((3,))
        y[free_axes] = x
        return y
    elif np.sum(free_axes) == 3:
        a, b = rng.random(2)
        th = np.arccos(2 * b - 1)
        phi = 2 * np.pi * a

        return np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi),
                         np.cos(th)])
    else:
        raise ValueError('free_axes must be a boolean array of length 3.')


def angular_momentum_range(J1, J2):
    return np.abs(J1 - J2), J1 + J2 + 1


def _d_q_matrix_element(J, F, m_F, Jp, Fp, m_Fp, q, I):
    return (-1) ** (F - m_F + J + I + Fp + 1) * np.sqrt((2 * F + 1) * (2 * Fp + 1)) * \
        wig3j(F, 1, Fp, -m_F, q, m_Fp) * wig6j(J, F, I, Fp, Jp, 1)


def mu_q_coupled(basis, gJ, gI, J, I):
    Fs, mFs = basis
    n = len(Fs) # TODO: get a two-state version of this
    mu_q = np.zeros((3, n, n))
    count = 0
    for ii, q in enumerate(range(-1, 2)):
        for index_1 in range(n):
            for index_2 in range(n):
                count += 1
                F1, F2, mF1, mF2 = Fs[index_1], Fs[index_2], mFs[index_1], mFs[index_2]
                if mF1 == mF2 + q:
                    mu_q[ii, index_1, index_2] -= (gJ * (-1) ** np.abs(F1 - mF1)
                                                   * wig3j(F1, 1, F2, -mF1, q, mF2)
                                                   * np.sqrt((2 * F2 + 1) * (2 * F1 + 1))
                                                   * (-1) ** (J + I + F2 + 1)
                                                   * wig6j(J, F2, I, F1, J, 1)
                                                   * np.sqrt(J * (J + 1) * (2 * J + 1)))

                    mu_q[ii, index_1, index_2] += (gI * (-1) ** np.abs(F1 - mF1)
                                                   * wig3j(F1, 1, F2, -mF1, q, mF2)
                                                   * np.sqrt((2 * F2 + 1) * (2 * F1 + 1))
                                                   * (-1) ** (J + I + F2 + 1)
                                                   * wig6j(I, F2, J, F1, I, 1)
                                                   * np.sqrt(I * (I + 1) * (2 * I + 1)))
    return mu_q



def hyperfine_correction(J, I, F, hf_coefs):
    """F can be a single number or an ndarray"""
    K = F * (F + 1) - I * (I + 1) - J * (J + 1)
    Ahf, Bhf, Chf = hf_coefs
    energy = Ahf * K / 2
    if Bhf != 0:
        energy += Bhf * (1.5 * K * (K + 1) - 2 * I * (I + 1) * J * (J + 1)) / (
                4 * I * (2 * I - 1) * J * (2 * J - 1))
    if Chf != 0:
        energy += Chf * (5 * K ** 2 * (K / 4 + 1)
                         + K * (I * (I + 1) + J * (J + 1) + 3 - 3 * I * (
                        I + 1) * J * (
                                        J + 1))
                         - 5 * I * (I + 1) * J * (J + 1)) / (
                          I * (I - 1) * (2 * I - 1) * J * (J - 1) * (2 * J - 1))
    return energy


def get_state_basis(state, Fs_sorted):
    mFs = []
    Fs = []
    for F in Fs_sorted:
        mFs += state.substates[F]
        Fs += [F] * len(state.substates[F])
    return Fs, mFs


# endregion

class Progressbar(object):
    def __init__(self, decimals=1, fill='█', prefix='Progress:',
                 suffix='', time_remaining_prefix=' time left', length=30,
                 update_rate=0.5):
        self.tic = time.time()
        self.decimals = decimals
        self.fill = fill
        self.length = length
        self.prefix = prefix
        self.suffix = suffix
        self.time_remaining_prefix = time_remaining_prefix
        self.finished = False
        self.max_written_length = 0
        self.last_update = 0.
        self.update_rate = update_rate

    def format_time(self, tic_toc):
        # Print a final time of completion
        if tic_toc > 3600:
            time_str = "%d:%02d:%02d" % ((tic_toc) / 3600.0,
                                         ((tic_toc) / 60.0) % 60.0,
                                         (tic_toc) % 60.0)
        elif tic_toc > 60:
            time_str = "%d:%02d" % ((tic_toc) / 60.0,
                                    (tic_toc) % 60.0)
        else:
            time_str = "%.2f s" % (tic_toc)

        return time_str

    def print_string(self, string1):
        # Update the maximum length of string written:
        self.max_written_length = max(self.max_written_length, len(string1))
        pad = ''.join([' '] * (self.max_written_length - len(string1)))
        print(string1 + pad, end='\r')

    def update(self, percentage):
        toc = time.time()
        if percentage > 0 and percentage < 1 and (toc - self.last_update) > self.update_rate:
            percent = ("{0:." + str(self.decimals) + "f}").format(100 * percentage)
            filledLength = int(self.length * percentage)
            bar = self.fill * filledLength + '-' * (self.length - filledLength)

            remaining_time = (1 - percentage) * ((toc - self.tic) / percentage)
            if remaining_time > 0:
                time_str = self.format_time(remaining_time)
            else:
                time_str = "0.00 s"
            self.print_string('%s |%s| %s%%%s;%s: %s' %
                              (self.prefix, bar, percent, self.suffix,
                               self.time_remaining_prefix, time_str))
            self.last_update = toc
        elif percentage >= 1:
            if not self.finished:
                self.finished = True
                time_str = self.format_time(toc - self.tic)
                self.print_string('Completed in %s.' % time_str)
                print()


class BaseForceProfile:
    """
    Base force profile

    The force profile object stores all the calculated quantities created by
    the governingeq.generate_force_profile() method.  It has the following
    attributes:

    Attributes
    ----------
    R : array_like, shape (3, ...)
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
        pylcp.laserBeams object used to create the governing equation.
    Neq : array_like
        Equilibrium population found.
    """

    def __init__(self, R, V, laserBeams, hamiltonian: Hamiltonian):
        if not isinstance(R, np.ndarray):
            R = np.array(R)
        if not isinstance(V, np.ndarray):
            V = np.array(V)

        if R.shape[0] != 3 or V.shape[0] != 3:
            raise TypeError('R and V must have first dimension of 3.')

        self.R = copy.copy(R)
        self.V = copy.copy(V)

        if hamiltonian is None:
            self.Neq = None
        else:
            self.Neq = np.zeros(R.shape[1:] + (hamiltonian.n,))

        self.f = {}
        for key in laserBeams:
            self.f[key] = np.zeros(R.shape + (len(laserBeams[key].beam_vector),))

        self.f_mag = np.zeros(R.shape)

        self.F = np.zeros(R.shape)

    def store_data(self, ind, Neq, F, F_laser, F_mag):
        if not Neq is None:
            self.Neq[ind] = Neq

        for jj in range(3):
            # self.f[(jj,) + ind] = f[jj]
            self.F[(jj,) + ind] = F[jj]
            for key in F_laser:
                self.f[key][(jj,) + ind] = F_laser[key][jj]

            self.f_mag[(jj,) + ind] = F_mag[jj]


def add_conventional_MOT_laser_kvecs_and_pols():
    rot_mat = Rotation.from_euler('ZYZ', [0, 0, 0]).as_matrix()

    kvecs = [np.array([1., 0., 0.]), np.array([-1., 0., 0.]),
             np.array([0., 1., 0.]), np.array([0., -1., 0.]),
             np.array([0., 0., 1.]), np.array([0., 0., -1.])]
    pol = PolarizationEnum.SIGMA_MINUS
    opp = pol.opposite()
    pols = [opp,opp,opp,opp, pol, pol]
    lasers = []
    for kvec, _pol in zip(kvecs, pols):
        lasers.append((rot_mat @ kvec, _pol))
    return lasers


def sort_float_then_string(numbers, strings):
    energy_label_pairs = np.asarray([(numbers[i], strings[i]) for i in range(len(numbers))],
                                    dtype=[('energy', float), ('label', 'S10')])
    return np.argsort(energy_label_pairs, order=['energy', 'label'])
