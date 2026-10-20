import os
import tempfile
import numpy as np
import pytest
from scipy.constants import c, h

import pylcp
from pylcp.atom.dataframe import DataFrame, PredefinedAtom
from pylcp.typing import HyperfineKey, HFTransitionKey, TransitionKey, PolarizationEnum
from pylcp.hamiltonian import Hamiltonian
from pylcp.fields import LaserBeams


class TestDataFrameBasics:
    def test_create_from_predefined_atoms(self):
        """Test instantiation of DataFrame from various predefined atoms."""
        # Rb87: I = 1.5, 5S1/2 (g), 5P1/2 (e1), 5P3/2 (e2)
        rb_frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        assert rb_frame.I == 1.5
        assert "g" in rb_frame.fine_states
        assert "e1" in rb_frame.fine_states
        assert "e2" in rb_frame.fine_states
        assert rb_frame.get_state_label(0) == "g"

        # Cs133: I = 3.5
        cs_frame = DataFrame.create_from_atom(PredefinedAtom.Cs133)
        assert cs_frame.I == 3.5
        assert "g" in cs_frame.fine_states
        assert cs_frame.get_state_label(0) == "g"

        # Li6: I = 1.0
        li_frame = DataFrame.create_from_atom(PredefinedAtom.Li6)
        assert li_frame.I == 1.0
        assert "g" in li_frame.fine_states

    def test_hyperfine_energy_splitting(self):
        """Test that DataFrame.hf_energy computes the known hyperfine splitting in Rb87 ground state."""
        rb_frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        e_f1 = rb_frame.hf_energy(HyperfineKey("g", 1.0))
        e_f2 = rb_frame.hf_energy(HyperfineKey("g", 2.0))
        # Ground state 5S1/2 has splitting ~ 6.83468261 GHz * h * c / 1e2 or in Hz
        diff_hz = np.abs(e_f2 - e_f1)
        expected_splitting_hz = 6.83468261090429e9
        # Relative error should be within 0.1%
        assert np.isclose(diff_hz, expected_splitting_hz, rtol=1e-3)

    def test_remove_states(self):
        """Test removing fine states and individual hyperfine manifolds."""
        rb_frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        assert "e1" in rb_frame.fine_states
        assert TransitionKey("g", "e1") in rb_frame.transitions

        # Remove e1 fine state
        rb_frame.remove_state("e1")
        assert "e1" not in rb_frame.fine_states
        assert TransitionKey("g", "e1") not in rb_frame.transitions

        # Remove specific hyperfine state F=0 from e2 (which has F=0, 1, 2, 3)
        e2_state = rb_frame.fine_states["e2"]
        assert 0.0 in e2_state.substates
        assert len(e2_state.substates[0.0]) > 0

        rb_frame.remove_state("e2", F=0.0)
        assert len(e2_state.substates[0.0]) == 0


class TestDataFrameHamiltonianAndLasers:
    def test_fine_structure_block_mode(self):
        """Test Hamiltonian and Laser generation when all active hyperfine transitions are coupled."""
        frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        g = frame.get_state_label(0)
        e2 = frame.get_state_label(2)
        frame.remove_state("e1")

        # Add conventional MOT beams for repump (F=1 -> F'=2) and cooling (F=2 -> F'=3)
        frame.add_conventional_MOT_lasers(g, e2, 1, 2, delta=0.0, intensity=0.01, delta_in_Gamma=True)
        frame.add_conventional_MOT_lasers(g, e2, 2, 3, delta=-2.0, intensity=1.0, delta_in_Gamma=True)

        # By default, add_laser enables all active hyperfine pairs in the freq_group,
        # so custom hyperfine coupling is False
        assert not frame._has_custom_hyperfine_coupling()

        ham = frame._hamiltonian()
        assert isinstance(ham, Hamiltonian)
        assert ham.state_labels == ["g", "e2"]

        # Number of states: ground has F=1 (3) + F=2 (5) = 8
        # Excited e2 has F=0 (1) + F=1 (3) + F=2 (5) + F=3 (7) = 16
        # Total = 24 states
        assert ham.ns == [8, 16]
        assert ham.n == 24

        lasers = frame._lasers()
        assert TransitionKey(g, e2) in lasers
        assert isinstance(lasers[TransitionKey(g, e2)], LaserBeams)
        # 6 beams per MOT set x 2 freq groups = 12 beams
        assert len(lasers[TransitionKey(g, e2)].beam_vector) == 12

    def test_hyperfine_block_mode(self):
        """Test Hamiltonian and Laser generation when a strict subset of hyperfine transitions is coupled."""
        frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        g = frame.get_state_label(0)
        e2 = frame.get_state_label(2)
        frame.remove_state("e1")

        # Add a laser targeting ONLY specific hyperfine transitions
        # by manually setting enabled_transitions to a strict subset
        key1, key2 = HyperfineKey(g, 2.0), HyperfineKey(e2, 3.0)
        hf_trans = HFTransitionKey(key1, key2)
        fine_trans = TransitionKey(g, e2)

        frame.add_laser(g, e2, 2, 3, delta=0.0, kvec=np.array([0, 0, 1]),
                        pol=PolarizationEnum.SIGMA_PLUS, intensity=1.0)

        # Restrict the enabled transitions of this laser frequency group to only (g_F2 -> e2_F3)
        fg = list(frame.lasers[fine_trans].freq_groups.values())[0]
        fg.enabled_transitions = [hf_trans]

        # Now custom hyperfine coupling is True
        assert frame._has_custom_hyperfine_coupling()

        ham = frame._hamiltonian()
        assert isinstance(ham, Hamiltonian)
        # Blocks must be separate hyperfine manifolds: g_F1, g_F2, e2_F0, e2_F1, e2_F2, e2_F3
        assert "g_F1" in ham.state_labels
        assert "g_F2" in ham.state_labels
        assert "e2_F3" in ham.state_labels

        lasers = frame._lasers()
        # In Hyperfine Block Mode, laser keys correspond to specific hyperfine transition keys
        assert TransitionKey("g_F2", "e2_F3") in lasers
        assert all(isinstance(k, TransitionKey) for k in lasers.keys())

    def test_obe_and_rateeq_creation(self):
        """Verify that frame.obe() and frame.rateeq() instantiate and evaluate without errors."""
        frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        g = frame.get_state_label(0)
        e2 = frame.get_state_label(2)
        frame.remove_state("e1")
        frame.add_conventional_MOT_lasers(g, e2, 2, 3, delta=-1.0, intensity=1.0, delta_in_Gamma=True)

        obe = frame.obe()
        assert obe is not None
        assert obe.hamiltonian is not None

        # Test evaluating the OBE evolution matrix at origin
        ev_mat = obe.full_OBE_ev(np.zeros(3), 0.0)
        assert ev_mat.shape == (obe.hamiltonian.n ** 2, obe.hamiltonian.n ** 2)
        import scipy.sparse as sp
        data = ev_mat.data if sp.issparse(ev_mat) else ev_mat
        assert not np.isnan(data).any()

        rateeq = frame.rateeq()
        assert rateeq is not None
        assert rateeq.hamiltonian is not None

    def test_save_and_load_roundtrip(self):
        """Test DataFrame pickle serialization and reloading."""
        frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        frame.remove_state("e1")
        frame.add_conventional_MOT_lasers("g", "e2", 2, 3, delta=-1.5, intensity=2.0)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            frame.save(tmp_path)
            loaded_frame = DataFrame.load_from_file(tmp_path)

            assert loaded_frame.I == frame.I
            assert loaded_frame.gI == frame.gI
            assert set(loaded_frame.fine_states.keys()) == set(frame.fine_states.keys())
            assert set(loaded_frame.transitions.keys()) == set(frame.transitions.keys())
            assert len(loaded_frame.lasers) == len(frame.lasers)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_laser_freq_group_transition_toggles(self):
        """Test enable_transition, disable_transition, and enable_single_transition methods on LaserFreqGroup."""
        from pylcp.atom.dataframe_internals import LaserFreqGroup
        from pylcp.typing import HyperfineKey, HFTransitionKey, TransitionKey

        t_key = TransitionKey("g", "e2")
        hft1 = HFTransitionKey(HyperfineKey("g", 1), HyperfineKey("e2", 2))
        hft2 = HFTransitionKey(HyperfineKey("g", 2), HyperfineKey("e2", 2))
        hft3 = HFTransitionKey(HyperfineKey("g", 1), HyperfineKey("e2", 1))

        fg = LaserFreqGroup(freq=1e14, transition=t_key, enabled_transitions=[hft1, hft2])
        assert len(fg.enabled_transitions) == 2
        assert hft1 in fg.enabled_transitions
        assert hft2 in fg.enabled_transitions

        # disable_transition
        fg.disable_transition(hft1)
        assert len(fg.enabled_transitions) == 1
        assert hft1 not in fg.enabled_transitions
        assert hft2 in fg.enabled_transitions

        # disable non-existent transition (no-op, no error)
        fg.disable_transition(hft1)
        assert len(fg.enabled_transitions) == 1

        # enable_transition
        fg.enable_transition(hft3)
        assert len(fg.enabled_transitions) == 2
        assert hft3 in fg.enabled_transitions

        # enable already enabled transition (no duplicate)
        fg.enable_transition(hft3)
        assert len(fg.enabled_transitions) == 2

        # enable_single_transition
        fg.enable_single_transition(hft1)
        assert fg.enabled_transitions == [hft1]

