import importlib
import numpy as np
import pytest

upstream_atom = importlib.import_module("tests.vendor.upstream_pylcp.atom")
fork_atom = importlib.import_module("pylcp.atom")

def test_state_comparison():
    """Compare atom.state object between upstream and fork."""
    up_state = upstream_atom.state(n=5, S=0.5, L=1, J=1.5, lam=780e-9, tau=26e-9, gJ=1.33)
    fk_state = fork_atom.state(n=5, S=0.5, L=1, J=1.5, lam=780e-9, tau=26e-9, gJ=1.33)
    
    assert up_state.energy == pytest.approx(fk_state.energy)
    assert up_state.gamma == pytest.approx(fk_state.gamma)
    assert up_state.gammaHz == pytest.approx(fk_state.gammaHz)
    assert up_state.gJ == pytest.approx(fk_state.gJ)

def test_alkali_atom_comparison():
    """Compare atom('87Rb') species constants between upstream and fork."""
    up_rb = upstream_atom.atom("87Rb")
    fk_rb = fork_atom.atom("87Rb")
    
    assert up_rb.I == pytest.approx(fk_rb.I)
    assert up_rb.gI == pytest.approx(fk_rb.gI)
    assert up_rb.mass == pytest.approx(fk_rb.mass)
    assert len(up_rb.state) == len(fk_rb.state)
    
    for s_up, s_fk in zip(up_rb.state, fk_rb.state):
        assert s_up.energy == pytest.approx(s_fk.energy)
        assert s_up.gJ == pytest.approx(s_fk.gJ)
