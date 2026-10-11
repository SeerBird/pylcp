import os
import numpy as np
import pytest

import pylcp.atom as fork_atom

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_state_comparison():
    """Compare atom.state object against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "atom_ref.npz"))
    fk_state = fork_atom.State(n=5, S=0.5, L=1, J=1.5, lam=780e-9, tau=26e-9, gJ=1.33)

    assert fk_state.energy == pytest.approx(float(ref["state_energy"]))
    assert fk_state.gamma == pytest.approx(float(ref["state_gamma"]))
    assert fk_state.gammaHz == pytest.approx(float(ref["state_gammaHz"]))
    assert fk_state.gJ == pytest.approx(float(ref["state_gJ"]))


def test_alkali_atom_comparison():
    """Compare atom('87Rb') species constants against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "atom_ref.npz"))
    fk_rb = fork_atom.Atom("87Rb")

    assert fk_rb.I == pytest.approx(float(ref["rb_I"]))
    assert fk_rb.gI == pytest.approx(float(ref["rb_gI"]))
    assert fk_rb.mass == pytest.approx(float(ref["rb_mass"]))
    assert len(fk_rb.states) == len(ref["rb_states_energy"])

    for s_fk, e_ref, gj_ref in zip(fk_rb.states, ref["rb_states_energy"], ref["rb_states_gJ"]):
        assert s_fk.energy == pytest.approx(float(e_ref))
        assert s_fk.gJ == pytest.approx(float(gj_ref))
