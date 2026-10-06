import numpy as np
import scipy.sparse as sparse
import pytest

def reference_loop_build_coherent_ev_submatrix(n, H):
    """Reference triple-loop implementation from original upstream pylcp."""
    ev_mat = np.zeros((n**2, n**2), dtype='complex128')
    for ii in range(n):
        for jj in range(n):
            for kk in range(n):
                ev_mat[ii + jj*n, ii + kk*n] += 1j * H[kk, jj]
                ev_mat[ii + jj*n, kk + jj*n] -= 1j * H[ii, kk]
    return ev_mat

def vectorized_build_coherent_ev_submatrix(n, H, use_sparse=False):
    """Vectorized Kronecker product implementation."""
    if use_sparse:
        In = sparse.eye(n, dtype='complex128', format='csr')
        H_sp = sparse.csr_matrix(H)
        return 1j * (sparse.kron(H_sp.T, In, format='csr') - sparse.kron(In, H_sp, format='csr'))
    else:
        In = np.eye(n, dtype='complex128')
        return 1j * (np.kron(H.T, In) - np.kron(In, H))

@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_coherent_ev_submatrix_vectorization_equality(n):
    """Verify vectorized Kronecker product matches reference loop output exactly."""
    np.random.seed(42)
    H = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    ev_loop = reference_loop_build_coherent_ev_submatrix(n, H)
    ev_vec_dense = vectorized_build_coherent_ev_submatrix(n, H, use_sparse=False)
    ev_vec_sparse = vectorized_build_coherent_ev_submatrix(n, H, use_sparse=True).toarray()

    np.testing.assert_allclose(ev_vec_dense, ev_loop, atol=1e-14)
    np.testing.assert_allclose(ev_vec_sparse, ev_loop, atol=1e-14)
