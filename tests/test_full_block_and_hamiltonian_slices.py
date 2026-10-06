import numpy as np
import pytest
from pylcp.typing import TransitionKey
from pylcp.hamiltonian import Hamiltonian, FullBlock, H0Block, VectorBlock

def test_full_block_offsets_and_slices():
    fb = FullBlock(n=3, m=5, noffset=2, moffset=6)
    assert fb.row_slice == slice(2, 5)
    assert fb.col_slice == slice(6, 11)
    assert fb.state_slices == (slice(2, 5), slice(6, 11))
    assert fb.is_diagonal is False
    assert fb.is_empty is True

def test_hamiltonian_slice_helpers():
    H0_g = np.zeros((3, 3))
    H0_e = np.zeros((5, 5))
    muq_g = np.zeros((3, 3, 3))
    muq_e = np.zeros((3, 5, 5))
    dq = np.zeros((3, 3, 5))

    H = Hamiltonian(H0_g, H0_e, muq_g, muq_e, dq, mass=87.0)

    assert H.n == 8
    assert H.get_state_slice('g') == slice(0, 3)
    assert H.get_state_slice('e') == slice(3, 8)

    key_ge = TransitionKey('g', 'e')
    key_eg = TransitionKey('e', 'g')
    key_gg = TransitionKey('g', 'g')
    key_ee = TransitionKey('e', 'e')

    assert H.get_vec_slice(key_ge) == (slice(0, 3), slice(3, 8))
    assert H.get_vec_slice(key_gg) == (slice(0, 3), slice(0, 3))
    assert H.get_vec_slice(key_ee) == (slice(3, 8), slice(3, 8))

    with pytest.raises(KeyError, match="increasing energy order"):
        H.get_vec_slice(key_eg)

    idx_gg = H.get_vec_indices(key_gg)
    idx_ge = H.get_vec_indices(key_ge)
    idx_ee = H.get_vec_indices(key_ee)

    assert len(idx_gg) == 9
    assert len(idx_ge) == 15
    assert len(idx_ee) == 25

    all_indices = np.concatenate([idx_gg, idx_ge, idx_ee])
    assert len(all_indices) == 49
    assert len(np.unique(all_indices)) == 49
    assert np.all((all_indices >= 0) & (all_indices < 64))
