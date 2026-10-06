import numpy as np

from .common import spherical2cart
from .typing import TransitionKey


# region blocks
class H0Block:  # TODO: consider only storing a single dimension?
    def __init__(self, label: str, M: np.ndarray):
        """assumes M is square, validation upstream"""
        if len(M.shape) != 2 or M.shape[0] != M.shape[1]:
            raise ValueError('H_0 must be a square 2D matrix')
        self.label = label
        self.matrix = M
        self.parameters = {}

        self.n = M.shape[0]

    def return_block_in_place(self, i, j, N):
        super_M = np.zeros((N, N), dtype='complex128')
        super_M[i:i + self.n, j:j + self.n] = self.matrix

        return super_M

    @property
    def diagonal(self):
        M = self.matrix
        if M.shape[0] == M.shape[1]:
            return np.count_nonzero(M - np.diag(np.diagonal(M))) == 0
        else:
            return False  # Cannot be diagonal, cause not square.

    def __repr__(self):
        return '(%s %dx)' % (self.label, self.n)

    def __str__(self):
        return '(%s %dx)' % (self.label, self.n)


class VectorBlock:
    def __init__(self, lower_label: str, upper_label: str, M: np.ndarray):
        if M.shape[0] != 3:
            raise ValueError('Vector block must have first dimension 3, as it is a vector')
        self.key:TransitionKey = TransitionKey(lower_label, upper_label)
        self.matrix = M
        self.parameters = {}

        self.n = M.shape[1]
        self.m = M.shape[2]

    @property
    def diagonal(self):
        M = self.matrix
        if M.shape[1] == M.shape[2]:
            return np.count_nonzero(M[1] - np.diag(np.diagonal(M[1]))) == 0
        else:
            return False  # Cannot be diagonal, cause not square.

    def return_block_in_place(self, i, j, N):
        super_M = np.zeros((3, N, N), dtype='complex128')
        super_M[:, i:i + self.n, j:j + self.m] = self.matrix
        return super_M

    def add_to_other_in_place(self, other, i, j, scale=1.):
        """Adds self.matrix to the other matrix in position i,j"""
        other[:, i:i + self.n, j:j + self.m] += scale * self.matrix


class MuQBlock(VectorBlock):
    def __init__(self, lower_label: str, upper_label: str, M: np.ndarray, muB: float):
        super().__init__(lower_label, upper_label, M)
        self.muB = muB


class DQBlock(VectorBlock):
    def __init__(self, lower_label: str, upper_label: str, M: np.ndarray, k: float, gamma: float):
        super().__init__(lower_label, upper_label, M)
        self.k = k
        self.gamma = gamma


class FullBlock:
    mu_q: MuQBlock | None
    d_q: DQBlock | None
    h_0: H0Block | None

    def __init__(self, n, m):
        self.n: int = n
        self.m: int = m

    def add_mu_q(self, mu_q: MuQBlock):
        if self.mu_q is not None:
            raise ValueError("mu_q already added")
        if self.n == mu_q.n and self.m == mu_q.m:
            self.mu_q = mu_q
        else:
            raise ValueError("New mu_q block shape doesn't match existing block shape")

    def add_d_q(self, d_q: DQBlock):
        if self.d_q is not None:
            raise ValueError("d_q already added")
        if self.n == d_q.n and self.m == d_q.m:
            self.d_q = d_q
        else:
            raise ValueError("New d_q block shape doesn't match existing block shape")

    def add_H_0(self, h_0: H0Block):
        if self.h_0 is not None:
            raise ValueError("H_0 already added")
        if self.n == h_0.n:
            self.h_0 = h_0
        else:
            raise ValueError("H_0 block matrix doesn't match block shape")


# endregion

# Next, define a Hamiltonian class to work out the internal states:


class Hamiltonian:
    """
    A representation of the Hamiltonian in blocks

    Diagonal blocks describe the internal structure of a manifold, and
    off-diagonal blocks describe how those manifolds are connected via laser
    Beams and the associated dipole matrix elements.  For most cases, the
    Hamiltonian is usually just a two level system.  In this case, the
    Hamiltonian can be initiated using the optional parameters below and the
    two manifolds are given the labels :math:`g` and :math:`e`.  You must
    supply the five positional arguments below in order to initiate the
    Hamiltonian in this way.

    For other constructions with more than two manifolds, one should construct
    the Hamiltonian using the `pylcp.hamiltonian.add_H_0_block()`,
    `pylcp.hamiltonian.add_mu_q_block()` and `pylcp.hamiltonian.add_d_q_block()`.
    Note that the order in which the diagonal blocks are added is the energy
    ordering of the manifolds, which is often obscured after the rotating
    wave approximation is taken (and implicitly assumed to be taken before
    construction of this Hamiltonian object).

    For more information, see the accompanying paper that describes the
    block nature of the Hamiltonian.

    Parameters
    ----------
    H0_g : array_like, shape (N, N), optional
        Ground manifold field-independent matrix
    H0_e : array_like, shape (M, M), optional
        Excited manifold field-independent matrix
    muq_g : array_like, shape (3, N, N), optional
        Ground manifold magnetic field-dependent component, in spherical basis.
    muq_e : array_like, shape (3, M, M), optional
        Excited manifold magnetic field-dependent component, in spherical basis.
    d_q : array_like, shape (3, N, M), optional
        Dipole operator that connects the ground and excited manifolds, in
        spherical basis.
    mass : float
        Mass of the atom or molecule
    muB : Bohr magneton
        Value of the Bohr magneton in the units of choice
    gamma : float
        Value of the decay rate associated with :math:`d_q`
    k : float
        Value of the wavevector associated with :math:`d_q`

    Attributes
    ----------
    ns : int
        Total number of states in the Hamiltonian
    state_labels : list of char
        Updated list of the state labels used in the Hamiltonian.
    blocks_by_keys : dict
        The laser keys dictionary translates laser pumping keys like `g->e` into
        block indices for properly extracting the associated :math:`d_q` matrix.
    """

    def __init__(self, *args, mass=1., muB=1, gamma=1., k=1):
        self.blocks: list[list[FullBlock]] = []
        self.state_labels = []
        self.ns = []
        self.block_count = 0
        self.n = 0
        self.blocks_by_keys: dict[TransitionKey, FullBlock] = {}
        self.d_q_by_keys: dict[TransitionKey, DQBlock]
        self.mass = mass

        if len(args) == 5:  # TODO: replace this with clear overloads
            self.add_H_0_block('g', args[0])
            self.add_H_0_block('e', args[1])
            self.add_mu_q_block('g', 'g', args[2], muB=muB)
            self.add_mu_q_block('e', 'e', args[3], muB=muB)

            self.add_d_q_block('g', 'e', args[4], gamma=gamma, k=k)
        elif len(args) > 2:
            raise ValueError('Unknown number of arguments.')
        elif len(args) > 0:
            raise NotImplementedError('Not yet programmed for %d arguments.' %
                                      len(args))

    def print_structure(self):
        """
        Print structure of the Hamiltonian
        """
        print(self.blocks)

    def set_mass(self, mass):
        """
        Sets the Hamiltonian's mass parameter

        Parameters
        ----------
        mass : float
            The mass of the atom or molecule of the Hamiltonian
        """
        self.mass = mass

    def __add_new_row_and_column(self):
        pass

    def add_H_0_block(self, state_label, H_0: np.ndarray):
        """
        Adds a new H_0 block to the hamiltonian

        Parameters
        ----------
        state_label : str
            Label for the manifold for which this new block applies
        H_0 : array_like, with shape (N, N)
            Square matrix that describes the field-independent part of this
            manifold's Hamiltonian.  This manifold must have N states.
        """
        new_block = H0Block(state_label, H_0.astype('complex128'))
        if state_label in self.state_labels:
            raise ValueError("This state label is already used")
        self.ns.append(new_block.n)
        self.state_labels.append(state_label)
        # region add new row and column, fill new column with empty FullBlock objects
        self.blocks.append([])
        self.block_count += 1
        for i in range(self.block_count):
            self.blocks[i][-1] = FullBlock(self.ns[i], new_block.n)
            self.blocks_by_keys[TransitionKey(self.state_labels[i], new_block.label)] \
                = FullBlock(new_block.n, self.ns[i])
        # endregion
        self.blocks[-1][-1].add_H_0(new_block)
        self.n += new_block.n
        self.__check_diagonal_submatrices_are_themselves_diagonal()

    def __validate_vector_block_and_get_full_block(self, vector_block: VectorBlock):
        lower_label, upper_label = vector_block.key
        if vector_block.matrix.shape[0] != 3:
            raise ValueError()
        if lower_label not in self.state_labels:
            raise ValueError(f'Lower label {lower_label} not in state labels')
        if upper_label not in self.state_labels:
            raise ValueError(f'Upper label {upper_label} not in state labels')
        block = self.blocks_by_keys[vector_block.key]
        if block.n != vector_block.n or block.m != vector_block.m:
            raise ValueError(
                "Added vector block doesn't match the shape of the existing block on the same labels")
        return block

    def add_mu_q_block(self, lower_label: str, upper_label: str, mu_q: np.ndarray, muB=1):
        r"""
        Add a new mu_q block to the Hamiltonian

        Parameters
        ----------
        lower_label
            lala
        upper_label
            lala
        mu_q
            lala
        muB
            lala
        """
        new_mu_q = MuQBlock(lower_label, upper_label, mu_q.astype('complex128'),muB)
        block = self.__validate_vector_block_and_get_full_block(new_mu_q)
        # TODO: consider making self.blocks upper-triangular and only ever adding the top bit
        block.add_mu_q(new_mu_q)
        self.__check_diagonal_submatrices_are_themselves_diagonal()

    def add_d_q_block(self, label1, label2, d_q: np.ndarray, k: float = 1., gamma: float = 1.):
        """
        Adds a new :math:`d_q` block to the hamiltonian to connect two
        manifolds together.

        Parameters
        ----------
        label1 : str
            Label for the first manifold to which this block applies
        label2 : str
            Label for the second manifold to which this block applies
        d_q : array_like, with shape (3, N, M)
            Matrix that describes the electric field dependent part of
            this dipole matrix element.  The first manifold must
        k : float, optional
            The mangitude of the k-vector for this $d_q$ block.  Default: 1
        gamma : float, optional
            The mangitude of the decay rate associated with this $d_q$ block.
            Default: 1
        """
        new_d_q = DQBlock(label1, label2, d_q.astype('complex128'), k, gamma)
        block = self.__validate_vector_block_and_get_full_block(new_d_q)
        block.add_d_q(new_d_q)
        self.d_q_by_keys[new_d_q.key] = new_d_q

    def make_full_matrices(self):
        """
        Returns the full matrices that define the Hamiltonian.

        Assembles the full Hamiltonian matrices from the stored block
        representation, and returns the Hamiltonian in the appropriate parts.
        For this function, :math:`n` is the number of states

        Returns
        -------
        H_0 : array_like, shape (n, n)
            The diagonal portion of the Hamiltonian
        mu_q : array_like, shape (3, N, N)
            The magnetic field dependent portion, in spherical basis.
        d_q : dictionary of array_like, shape (3, N, N)
            The electric field dependent portion, in spherical basis, arranged
            by keys that describe the manifolds connected by the specific
            :math:`d_q`.  This usually gets paired with :math:`E^*`
        d_q_star : dictionary of array_like, shape (3, N, N)
            The electric field dependent portion, in spherical basis, arranged
            by keys that describe the manifolds connected by the specific
            :math:`d_q`.  This usually gets paired with :math:`E`
        """
        # Initialize the field-independent component of the Hamiltonian.
        self.H_0 = np.zeros((self.n, self.n), dtype='complex128')
        self.mu_q = np.zeros((3, self.n, self.n), dtype='complex128')

        n = 0

        # First, return H_0 and mu_q:
        for diag_block in np.diag(self.blocks):
            if isinstance(diag_block, self.VectorBlock):
                diag_block.add_to_other_in_place(self.mu_q, n, n,
                                                 diag_block.parameters['mu_B'])
            elif isinstance(diag_block, self.H0Block):
                self.H_0 += diag_block.return_block_in_place(n, n, self.n)
                n += diag_block.n
            else:
                self.H_0 += diag_block[0].return_block_in_place(n, n, self.n)
                self.mu_q += (diag_block[1].parameters['mu_B'] *
                              diag_block[1].return_block_in_place(n, n, self.n))
                n += diag_block[0].n

        self.d_q_bare = {}

        # Next, return d_q:
        for ii in range(self.blocks.shape[0]):
            for jj in range(ii + 1, self.blocks.shape[1]):
                if not self.blocks[ii, jj] is None:
                    key = TransitionKey(self.state_labels[ii], self.state_labels[jj])
                    nstart = int(np.sum(self.ns[:ii]))
                    mstart = int(np.sum(self.ns[:jj]))
                    self.d_q_bare[key] = self.blocks[ii, jj].return_block_in_place(nstart, mstart,
                                                                                   self.n)

        self.d_q_star = {}
        for key in self.d_q_bare.keys():
            self.d_q_star[key] = np.zeros(self.d_q_bare[key].shape, dtype='complex128')
            for kk in range(3):
                self.d_q_star[key][kk] = np.conjugate(self.d_q_bare[key][kk].T)

        # Finally, put together the full d_q, irrespective of laser beam key:
        self.d_q = np.zeros((3, self.n, self.n), dtype='complex128')
        for key in self.d_q_bare.keys():
            self.d_q += self.d_q_bare[key] + self.d_q_star[key]

        # Make Cartesian coordinate copies.
        self.mu = spherical2cart(self.mu_q)
        self.d = spherical2cart(self.d_q)

        return self.H_0, self.mu_q, self.d_q_bare, self.d_q_star

    def return_full_H(self, Eq, Bq):
        """
        Assemble the block diagonal Hamiltonian into a single matrix

        Parameters
        ----------
        Eq : array_like or dictionary of array_like
            The electric field(s) driving transitions between manifolds,
            each expressed in the spherical basis.  Each electric field
            driving a transition between manifolds needs to specified with the
            correct key in the dictionary.  For example, for a two-manifold
            Hamiltonian with manifold labels `g` and `e`, the dictionary should
            contain a single entry with `g->e`.  If the electric field is
            given as a single array_like, it is assumed to drive the `g->e`
            transition.
        Bq : array_like, shape (3,)
            The magnetic field in spherical basis.

        Returns
        -------
        H : array_like
            The full Hamiltonian matrix
        """
        if not hasattr(self, 'H_0'):
            self.make_full_matrices()

        H = self.H_0 - np.tensordot(self.mu_q, np.conjugate(Bq), axes=(0, 0))

        if isinstance(Eq, list) or isinstance(Eq, np.ndarray):
            Eq = {TransitionKey('g', 'e'): Eq}  # Promote to a dictionary.

        for key, val in Eq.items():
            for ii, q in enumerate([-1.0, 0.0, 1.0]):
                H -= (0.5 * (-1.) ** q * self.d_q_bare[key][ii] * val[2 - ii] +
                      0.5 * (-1.) ** q * self.d_q_star[key][ii] * np.conjugate(val[2 - ii]))

        return H

    def __check_diagonal_submatrices_are_themselves_diagonal(self):
        self.diagonal = np.zeros(self.block_count, dtype='bool')
        # TODO: fix this, maybe only do this before execution? or property? or whatever
        for ii, diag_block in enumerate(np.diag(self.blocks)):
            if isinstance(diag_block, tuple):
                self.diagonal[ii] = (diag_block[0].diagonal and diag_block[1].diagonal)
            else:
                self.diagonal[ii] = diag_block.diagonal

    def diag_static_field(self, B):
        r"""
        Block diagonalize at a specified magnetic field

        This function diagonalizes the Hamiltonian's diagonal blocks separately
        based on the value of the static magnetic field :math:`B`, and then
        rotates the :math:`d_q`s into the new bases. This is necessary for the
        rate equations, which always assume that :\math:`B` sets the quantization
        axis, and they rotate the coordinate system appropriately, so we only
        ever need to consider the z-component of the field.

        Parameters
        ----------
        B : float
            The magnetic field value at which to diagonalize.  It is always
            assumed to be along the :math:`\hat{z}` direction.

        Returns:
        H : pylcp.hamiltonian
            A block-structured Hamiltonian with diagonal elemented diagonalized
            and :math:`d_q` objects rotated
        """
        # Now we get to the meat of it:
        if not (isinstance(B, float) or isinstance(B, int)):
            raise ValueError('diag_static_field: the field should be given ' +
                             'by a single number, the magnitude (assumed ' +
                             'to be along z).')

        # If it does not already exist, make an empty Hamiltonian that has
        # the same dimensions as this one.
        if not hasattr(self, 'rotated_hamiltonian'):
            self.rotated_hamiltonian = Hamiltonian()
            for ii, block in enumerate(np.diagonal(self.blocks)):
                self.rotated_hamiltonian.add_H_0_block(
                    self.state_labels[ii],
                    np.zeros((self.ns[ii], self.ns[ii]), dtype='complex128')
                )
            for ii, block_row in enumerate(self.blocks):
                for jj, block in enumerate(block_row):
                    if jj > ii:
                        if not block is None:
                            self.rotated_hamiltonian.add_d_q_block(
                                self.state_labels[ii], self.state_labels[jj],
                                block.matrix,
                            )

        # Have we previously generated a set of transformation matrices?
        if not hasattr(self, 'U'):
            self.U = np.empty((self.blocks.shape[0],), dtype=object)
            # Now, go through all of the diagonal elements:
            for ii, diag_block in enumerate(np.diag(self.blocks)):
                # Make a transformation matrix that is boring.  We'll overwrite
                # it later if it gets interesting.
                self.U[ii] = np.eye(self.ns[ii])

        # Now, are any of the diagonal submatrices not diagonal?
        if not np.all(self.diagonal) or B < 0:
            # If so, go through all of the diagonal elements:
            for ii, diag_block in enumerate(np.diag(self.blocks)):
                # It isn't? Diagonalize it:
                if not self.diagonal[ii] or B < 0:
                    if isinstance(diag_block, tuple):
                        H = (diag_block[0].matrix - B * diag_block[1].matrix[1])
                    elif isinstance(diag_block, self.VectorBlock):
                        H = -B * diag_block.matrix[1]
                    else:
                        H = diag_block.matrix

                    # Diagonalize at this field:
                    Es, self.U[ii] = np.linalg.eig(H)

                    # Sort the  output, store the transformation matrix.
                    ind_e = np.argsort(Es)
                    Es = Es[ind_e]
                    self.U[ii] = self.U[ii][:, ind_e]

                    # Check to make sure the diganolization resulted in only real
                    # components, then build the matrix with the eigenvalues and
                    # go
                    if np.allclose(np.imag(Es), 0.):
                        self.rotated_hamiltonian.blocks[ii, ii].matrix = np.diag(np.real(Es))
                    else:
                        raise ValueError('You broke the Hamiltonian!')
                else:  # It is diagonal:
                    if isinstance(diag_block, tuple):
                        self.rotated_hamiltonian.blocks[ii, ii].matrix = \
                            diag_block[0].matrix - B * diag_block[1].matrix[1]
                    elif isinstance(diag_block, self.VectorBlock):
                        self.rotated_hamiltonian.blocks[ii, ii].matrix = \
                            -B * diag_block.matrix[1]
                    else:
                        self.rotated_hamiltonian.blocks[ii, ii].matrix = \
                            diag_block.matrix

            # Now, rotate the d_q:
            for ii in range(self.blocks.shape[0]):
                for jj in range(ii + 1, self.blocks.shape[1]):
                    if (not self.blocks[ii, jj] is None) and (
                            not self.diagonal[ii] or not self.diagonal[jj] or B < 0):
                        for kk in range(3):
                            self.rotated_hamiltonian.blocks[ii, jj].matrix[kk] = \
                                self.U[ii].T @ self.blocks[ii, jj].matrix[kk] @ self.U[jj]

                            self.rotated_hamiltonian.blocks[jj, ii].matrix[kk] = \
                                np.conjugate(self.rotated_hamiltonian.blocks[ii, jj].matrix[kk].T)

                            if (self.rotated_hamiltonian.blocks[ii, jj].matrix.shape !=
                                    self.blocks[ii, jj].matrix.shape):
                                raise ValueError("Rotataed d_q not the same size as original.")
        else:
            # We are already diagonal, so all we have to do is change the
            # eigenvalues.
            for ii, diag_block in enumerate(np.diag(self.blocks)):
                if isinstance(diag_block, tuple):
                    self.rotated_hamiltonian.blocks[ii, ii].matrix = \
                        np.real(diag_block[0].matrix - B * diag_block[1].matrix[1])
                elif isinstance(diag_block, self.VectorBlock):
                    self.rotated_hamiltonian.blocks[ii, ii].matrix = \
                        np.real(-B * diag_block.matrix[1])
                else:
                    self.rotated_hamiltonian.blocks[ii, ii].matrix = \
                        np.real(diag_block.matrix)

        return self.rotated_hamiltonian

    def diag_H_0(self, B0):
        pass
