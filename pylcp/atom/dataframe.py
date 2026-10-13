from __future__ import annotations

import logging
import pickle
from enum import Enum
from os import PathLike
from typing import TypeAlias, Callable

import numpy as np
from scipy.constants import c, h

from .atom import State as Pylcp_state, Atom
from pylcp.hamiltonian import Hamiltonian
from pylcp.obe import OBE
from pylcp.rateeq import RateEq
from pylcp.fields import MagField, LaserBeams, InfinitePlaneWaveBeam
from pylcp.atom.dataframe_internals import StateData, TransitionData, LaserTransitionGroup, \
    LaserDisplayData, LaserFreqGroup, LaserData
from pylcp.typing import HyperfineKey, HFTransitionKey, TransitionKey, Polarization, Vector3D
from pylcp.util import sort_float_then_string, hyperfine_correction, get_state_basis, \
    mu_q_coupled, _d_q_matrix_element, add_conventional_MOT_laser_kvecs_and_pols

logger: logging.Logger = logging.getLogger(__name__)

MagneticFieldObject: TypeAlias = (MagField
                                  | Callable[[Vector3D, float], Vector3D]
                                  | Callable[[Vector3D], Vector3D]
                                  | Vector3D)


class PredefinedAtom(Enum):
    Li6 = "Li6"
    Li7 = "Li7"
    Na23 = "Na23"
    K39 = "K39"
    K40 = "K40"
    K41 = "K41"
    Rb85 = "Rb85"
    Rb87 = "Rb87"
    Cs133 = "Cs133"


class DataFrame:
    def __init__(self, I, gI):
        """
        :param I:
        :param gI:
        """
        self.I: float = I
        self.gI = gI
        self.fine_states: dict[str, StateData] = {}
        self.transitions: dict[TransitionKey, TransitionData] = {}
        self.lasers: dict[TransitionKey, LaserTransitionGroup] = {}
        self.laser_displays: dict[HFTransitionKey, dict[float, LaserDisplayData]] = {}
        self.magnetic_field: MagField = MagField(np.zeros(3))

    @staticmethod
    def load_from_file(path: PathLike | str) -> DataFrame:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path: PathLike | str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def get_state_label(self, energy_index: int):
        """
        :param energy_index: the index of the state in the list of all states sorted in ascending
         energy order, i.e. 0 for lowest-energy state, 1 for second-lowest, and so on
        :return: the label of the state
        """
        energies = np.asarray([state.energy for state in self.fine_states.values()])
        labels = np.asarray([state.label for state in self.fine_states.values()])
        sort = sort_float_then_string(energies, labels)
        return labels[sort][energy_index]

    # region get pylcp results
    def obe(self):
        ham = self._hamiltonian()
        lasers = self._lasers()
        return OBE(lasers, self.magnetic_field, ham)

    def rateeq(self):
        return RateEq(self._lasers(), self.magnetic_field, self._hamiltonian(),
                      include_mag_forces=False)

    # endregion
    # region building the dataframe
    @classmethod
    def create_from_atom(cls, atom: PredefinedAtom):
        atom_data = Atom(atom.value)
        states = atom_data.states
        frame = DataFrame(atom_data.I, atom_data.gI)
        if len(states) == 0:
            return frame  # This shouldn't happen, ever, but it's fine if it does
        elif len(states) == 1:
            labels = ['g']  # This shouldn't happen either, but it's fine
        elif len(states) == 2:
            labels = ['g', 'e']
        else:
            labels = ['g']
            for i in range(1, len(states)):
                labels.append(f"e{i}")
        # region add all fine structure states
        for i in range(len(states)):
            state: Pylcp_state = states[i]
            frame.add_fine_state(labels[i],
                                 state.energy * 1e2 * c,
                                 state.J,
                                 (state.Ahfs, state.Bhfs, state.Chfs),
                                 state.gJ)
        # endregion
        # region add all transitions
        for i in range(1, len(states)):
            state: Pylcp_state = states[i]
            excited_state = frame.fine_states[labels[i]]
            frame.add_transition(labels[0], labels[i], state.gammaHz)
        # endregion
        return frame

    def set_magnetic_field(self, magnetic_field: MagneticFieldObject):
        if isinstance(magnetic_field, MagField):
            self.magnetic_field = magnetic_field
        else:
            self.magnetic_field = MagField(magnetic_field)

    def add_fine_state(self, label: str, energy: float, J: float,
                       hf_coefs: tuple[float, float, float], gJ: float):
        fine_state = StateData(label, energy, self.I, J, hf_coefs, gJ)
        self.fine_states[label] = fine_state
        return fine_state

    def add_transition(self, label1, label2, gamma: float):
        key = TransitionKey(label1, label2)
        self.transitions[key] = TransitionData(gamma)
        self.lasers[key] = LaserTransitionGroup(key)

    def add_laser(self, label1, label2, F1, F2, delta, kvec,
                  pol: Polarization, intensity, delta_in_Gamma=False):
        """
        :param label1: lower state label
        :param label2: upper state label
        :param F1:
        :param F2:
        :param delta: in (global) gamma units, relative to the F1-F2 transition
        :param kvec: unit vector in spherical coords
        :param pol: +1 or -1, or an np.ndarray in spherical polar coordinates, or a callable
        :param intensity: normalized to the saturation intensity
        :param delta_in_Gamma:
        :return:
        """

        key1, key2 = HyperfineKey(label1, F1), HyperfineKey(label2, F2)
        hf_transition = HFTransitionKey(key1, key2)
        fine_transition = hf_transition.to_transition()
        transition_energy = self.hf_energy(key2) - self.hf_energy(key1)
        if delta_in_Gamma:
            delta *= self._principal_gamma_and_energy()[0]
        freq = transition_energy + delta
        if not fine_transition in self.lasers:
            raise ValueError()
        transition_group = self.lasers[fine_transition]
        if freq in transition_group.freq_groups:
            freq_group = transition_group.freq_groups[freq]
        else:
            enabled_transitions = []
            lower_state = self.fine_states[fine_transition.lower]
            upper_state = self.fine_states[fine_transition.upper]
            for hfkey1 in lower_state.hyperfine_keys():
                if not lower_state.substates[hfkey1.F]:
                    continue
                for hfkey2 in upper_state.hyperfine_keys():
                    if not upper_state.substates[hfkey2.F]:
                        continue
                    enabled_transitions.append(HFTransitionKey(hfkey1, hfkey2))

            freq_group = LaserFreqGroup(freq, fine_transition, enabled_transitions)
            transition_group.freq_groups[freq] = freq_group

        freq_group.add_laser(LaserData(freq, kvec, pol, intensity
                                       * self._saturation_intensity(hf_transition)))

    def add_conventional_MOT_lasers(self, label1, label2, F1, F2,
                                    delta, intensity, delta_in_Gamma=False):
        for kvec, pol in add_conventional_MOT_laser_kvecs_and_pols():
            self.add_laser(label1, label2, F1, F2, delta, kvec, pol, intensity, delta_in_Gamma)

    def add_laser_display(self, freqIndex: int, F_ground: float, F_excited: float,
                          ground_label="g", excited_label="e", upwards: bool = True) -> None:
        """

        :param freqIndex: the index of the frequency group, in ascending order.
        Example: if this is the lowest-frequency laser coupling this !fine! state pair,
        the index is 0, second lowest — 1, and so on.
        :param F_ground:
        :param F_excited:
        :param ground_label:
        :param excited_label:
        :param upwards: True if the display has the detuning indicator on the upper state,
         false if on the lower
        """
        keys = HFTransitionKey(HyperfineKey(ground_label, float(F_ground)),
                               HyperfineKey(excited_label, float(F_excited)))
        fine_transition = keys.to_transition()
        freqs = list(self.lasers[fine_transition].freqs())
        freq = np.sort(np.asarray(freqs))[freqIndex]
        self._add_laser_display_from_data(LaserDisplayData(freq, keys, upwards))

    def _add_laser_display_from_data(self, display_data: LaserDisplayData):
        keys = display_data.keys
        if keys in self.laser_displays:
            if display_data.freq in self.laser_displays[keys]:
                raise ValueError(
                    "A laser display of this laser energy on this pair of hf states already exists")
        self.laser_displays.setdefault(keys, {})[display_data.freq] = display_data

    def remove_state(self, label: str, F: float | None = None, mF: float | None = None):
        if F is None:  # removing a fine state
            fine_state = self.fine_states.pop(label)
            for transition_key in list(self.transitions.keys()):
                if fine_state.label in transition_key:
                    self.transitions.pop(transition_key)
            # TODO: remove lasers and laser displays involving this state
        elif mF is None:  # removing a hyperfine state
            fine_state = self.fine_states[label]
            fine_state.substates[F] = []
            # TODO: remove laser freqgroup couplings involving this state
        else:  # removing a magnetic state
            pass  # TODO: add magnetic state removal from dataframe

    # endregion
    # region results helpers
    def hf_energy(self, key: HyperfineKey):
        fine_state = self.fine_states[key.label]
        return fine_state.energy + hyperfine_correction(fine_state.J, self.I, key.F,
                                                        fine_state.hf_coefs)

    def _has_custom_hyperfine_coupling(self) -> bool:
        """Returns True if any laser frequency group couples to a strict subset
        of available active hyperfine transitions."""
        for fine_trans, laser_group in self.lasers.items():
            if fine_trans.lower not in self.fine_states or fine_trans.upper not in self.fine_states:
                continue
            state1 = self.fine_states[fine_trans.lower]
            state2 = self.fine_states[fine_trans.upper]

            active_Fs1 = [F for F, mFs in state1.substates.items() if len(mFs) > 0]
            active_Fs2 = [F for F, mFs in state2.substates.items() if len(mFs) > 0]
            full_hf_pairs = {
                HFTransitionKey(HyperfineKey(fine_trans.lower, F1),
                                HyperfineKey(fine_trans.upper, F2))
                for F1 in active_Fs1 for F2 in active_Fs2
            }
            if not full_hf_pairs:
                continue

            for fg in laser_group.freq_groups.values():
                if not fg.lasers:
                    continue
                enabled_set = set(fg.enabled_transitions)
                if enabled_set != full_hf_pairs:
                    return True
        return False

    def _hamiltonian(self):
        ham = Hamiltonian()
        ref_gamma: float = self._principal_gamma_and_energy()[0]
        labels = np.asarray(list(self.fine_states.keys()))
        energies = np.asarray([self.fine_states[label].energy for label in labels])
        sort = sort_float_then_string(energies, labels)
        labels = labels[sort]  # in the order the blocks should be added
        # TODO: consider calculating the matrix only once and THEN breaking it up into blocks.
        if not self._has_custom_hyperfine_coupling():
            # region Fine-Structure Block Mode (no custom laser hyperfine filtering)
            for state_label in labels:
                fine_state: StateData = self.fine_states[state_label]
                active_Fs = [F for F, mFs in fine_state.substates.items() if len(mFs) > 0]
                if not active_Fs:
                    continue
                Fs = np.asarray(active_Fs)
                sort = np.argsort(hyperfine_correction(fine_state.J, self.I, Fs,
                                                       fine_state.hf_coefs))
                Fs_sorted = Fs[sort]
                basis = get_state_basis(fine_state, Fs_sorted)
                Fs_vec, mFs_vec = basis
                n = len(Fs_vec)

                # Diagonal H_0 with hyperfine corrections
                H_0 = np.zeros((n, n), dtype=complex)
                for i in range(n):
                    H_0[i, i] = hyperfine_correction(fine_state.J, self.I, Fs_vec[i],
                                                     fine_state.hf_coefs) / ref_gamma
                ham.add_H_0_block(state_label, H_0)

                # mu_q full matrix (includes intra- and inter-hyperfine magnetic couplings)
                mu_q = mu_q_coupled(basis, fine_state.gJ, self.gI, fine_state.J, self.I)
                ham.add_mu_q_block(state_label, state_label, mu_q)  # TODO: off-diagonal!

            for traverse_label_pair in self.transitions.keys():
                label1, label2 = traverse_label_pair.lower, traverse_label_pair.upper
                if label1 not in self.fine_states or label2 not in self.fine_states:
                    continue
                state_1 = self.fine_states[label1]
                state_2 = self.fine_states[label2]
                active_Fs1 = np.sort([F for F, mFs in state_1.substates.items() if len(mFs) > 0])
                active_Fs2 = np.sort([F for F, mFs in state_2.substates.items() if len(mFs) > 0])
                if len(active_Fs1) == 0 or len(active_Fs2) == 0:
                    continue
                basis1 = get_state_basis(state_1, active_Fs1)
                basis2 = get_state_basis(state_2, active_Fs2)
                Fs1, mFs1 = basis1
                Fs2, mFs2 = basis2

                d_q = np.zeros((3, len(Fs1), len(Fs2)))
                J1, J2 = state_1.J, state_2.J
                I = self.I
                for index_1 in range(len(Fs1)):
                    for index_2 in range(len(Fs2)):
                        f1_val, f2_val = Fs1[index_1], Fs2[index_2]
                        mf1_val, mf2_val = mFs1[index_1], mFs2[index_2]
                        for q_index, q in enumerate((-1., 0., 1.)):
                            if mf2_val == mf1_val - q:
                                d_q[q_index, index_1, index_2] = _d_q_matrix_element(J1, f1_val,
                                                                                     mf1_val,
                                                                                     J2, f2_val,
                                                                                     mf2_val,
                                                                                     q, I)
                d_q *= np.sqrt(J2 * 2 + 1)
                transition_energy = np.abs(state_1.energy - state_2.energy)
                transition_gamma = self.transitions[traverse_label_pair].gamma
                # TODO: for extremely small transition gammas, figure out if rescaling Gamma' = Gamma/k,
                #  d_q' = d_q * sqrt(k), s' = k*s (for laser) works in handling the 'infinite s' issue
                ham.add_d_q_block(label1, label2, d_q,
                                  k=transition_energy / ref_gamma,
                                  gamma=transition_gamma / ref_gamma)
            return ham
            # endregion
        # region Hyperfine Block Mode (fallback when custom laser hyperfine filtering is present)
        hf_bases = {}
        # region H_0 & mu_q per hyperfine manifold
        for state_label in labels:
            fine_state: StateData = self.fine_states[state_label]
            active_Fs = [F for F, mFs in fine_state.substates.items() if len(mFs) > 0]
            if not active_Fs:
                continue
            Fs = np.asarray(active_Fs)
            sort = np.argsort(hyperfine_correction(fine_state.J, self.I, Fs,
                                                   fine_state.hf_coefs))
            Fs = Fs[sort]

            for idx, F in enumerate(Fs):
                block_name = str(HyperfineKey(state_label, F))
                basis = get_state_basis(fine_state, [F])
                # TODO: this reuses old code,
                #  make something that matches the new architecture cleaner
                hf_bases[HyperfineKey(state_label, F)] = (block_name, basis)

                # H_0 block (zeros as we're in the rotating frame of each hyperfine state)
                mF_count = len(fine_state.substates[F])
                ham.add_H_0_block(block_name, np.zeros((mF_count, mF_count)))

                # mu_q block
                mu_q = mu_q_coupled(basis, fine_state.gJ, self.gI, fine_state.J, self.I)
                ham.add_mu_q_block(block_name, block_name, mu_q)
        # endregion
        # region d_q between active hyperfine pairs
        for traverse_label_pair in self.transitions.keys():
            label1, label2 = traverse_label_pair.lower, traverse_label_pair.upper
            state_1 = self.fine_states[label1]
            state_2 = self.fine_states[label2]
            J1, J2 = state_1.J, state_2.J
            transition_energy = np.abs(state_1.energy - state_2.energy)
            transition_gamma = self.transitions[traverse_label_pair].gamma

            active_Fs1 = [F for F, mFs in state_1.substates.items() if len(mFs) > 0]
            active_Fs2 = [F for F, mFs in state_2.substates.items() if len(mFs) > 0]

            for F1 in active_Fs1:
                for F2 in active_Fs2:
                    if (HyperfineKey(label1, F1) not in hf_bases or
                            HyperfineKey(label2, F2) not in hf_bases):
                        continue
                    b1_name, basis1 = hf_bases[(label1, F1)]
                    b2_name, basis2 = hf_bases[(label2, F2)]
                    Fs1, mFs1 = basis1
                    Fs2, mFs2 = basis2

                    d_q = np.zeros((3, len(Fs1), len(Fs2)))
                    I = self.I
                    for index_1 in range(len(Fs1)):
                        for index_2 in range(len(Fs2)):
                            f1_val, f2_val, mf1_val, mf2_val = Fs1[index_1], Fs2[index_2], mFs1[
                                index_1], mFs2[index_2]
                            for q_index, q in enumerate((-1., 0., 1.)):
                                if mf2_val == mf1_val - q:
                                    d_q[q_index, index_1, index_2] = _d_q_matrix_element(J1, f1_val,
                                                                                         mf1_val,
                                                                                         J2, f2_val,
                                                                                         mf2_val,
                                                                                         q, I)
                    d_q *= np.sqrt(J2 * 2 + 1)
                    # TODO: for extremely small transition gammas, figure out if rescaling Gamma' = Gamma/k,
                    #  d_q' = d_q * sqrt(k), s' = k*s (for laser) works in handling the 'infinite s' issue
                    ham.add_d_q_block(b1_name, b2_name, d_q,
                                      k=transition_energy / ref_gamma,
                                      gamma=transition_gamma / ref_gamma)
        # endregion
        # endregion
        return ham

    def _lasers(self) -> dict[TransitionKey, LaserBeams]:
        lasers = {}
        laser_lists = {}
        ref_gamma, ref_energy = self._principal_gamma_and_energy()

        if not self._has_custom_hyperfine_coupling():
            # region Fine-Structure Block Mode
            for fine_trans, fine_transition_laser_group in self.lasers.items():
                if fine_trans.lower not in self.fine_states or fine_trans.upper not in self.fine_states:
                    continue
                fine_energy_diff = np.abs(self.fine_states[fine_trans.upper].energy -
                                          self.fine_states[fine_trans.lower].energy)
                beam_list = []
                for freq_group in fine_transition_laser_group.freq_groups.values():
                    freq = freq_group.freq
                    delta = freq - fine_energy_diff
                    if not freq_group.enabled_transitions:
                        continue
                    sat_intensity = self._saturation_intensity(
                        list(freq_group.enabled_transitions)[0])
                    for laser_data in freq_group.lasers:
                        beam_list.append(InfinitePlaneWaveBeam(laser_data.kvec * freq / ref_energy,
                                                               laser_data.pol,
                                                               laser_data.intensity / sat_intensity,
                                                               delta / ref_gamma))
                if beam_list:
                    lasers[fine_trans] = LaserBeams(beam_list)
            return lasers
            # endregion

        # region Hyperfine Block Mode (fallback)
        for fine_transition_laser_group in self.lasers.values():
            for freq_group in fine_transition_laser_group.freq_groups.values():
                freq = freq_group.freq
                for hf_transition in freq_group.enabled_transitions:
                    hf_key1, hf_key2 = hf_transition
                    sat_intensity = self._saturation_intensity(hf_transition)
                    delta = (freq - np.abs(self.hf_energy(hf_key2) - self.hf_energy(hf_key1)))
                    if hf_transition not in laser_lists:
                        laser_lists[hf_transition] = []
                    laser_list = laser_lists[hf_transition]
                    for laser_data in freq_group.lasers:
                        laser_list.append(InfinitePlaneWaveBeam(laser_data.kvec * freq / ref_energy,
                                                                laser_data.pol,
                                                                laser_data.intensity / sat_intensity,
                                                                delta / ref_gamma))
        for hf_transition in laser_lists:
            lasers[str(hf_transition)] = LaserBeams(laser_lists[hf_transition])
        return lasers

    def _principal_gamma_and_energy(self):
        """
        :return: the gamma and energy of the transition with the highest gamma
        """
        reference_gamma = 0
        reference_energy = None
        for label_pair in self.transitions:
            transition_gamma = self.transitions[label_pair].gamma
            if transition_gamma > reference_gamma:
                reference_gamma = transition_gamma
                reference_energy = (self.fine_states[label_pair[1]].energy -
                                    self.fine_states[label_pair[0]].energy)
        return reference_gamma, reference_energy

    def _saturation_intensity(self, hf_transition: HFTransitionKey):
        fine_transition = hf_transition.to_transition()
        energy = np.abs(self.fine_states[fine_transition.lower].energy -
                        self.fine_states[fine_transition.upper].energy)  # eV
        gamma = self.transitions[fine_transition].gamma  # Hz
        return (2 * np.pi ** 2 * h * energy ** 3 * gamma) / (3 * c ** 2)
    # endregion
