import numpy as np
import pytest
from scipy.constants import c, h, mu_0, physical_constants

import pylcp
from pylcp.atom.dataframe import DataFrame, PredefinedAtom
from pylcp.fields import (
    LaserBeam, InfinitePlaneWaveBeam,
    RFField,
    LaserBeams, MagField
)
from pylcp.hamiltonian import Hamiltonian, VectorBlock, DQBlock, MuQBlock
from pylcp.obe import OBE
from pylcp.rateeq import RateEq
from pylcp.typing import HyperfineKey, HFTransitionKey, TransitionKey, PolarizationEnum
from pylcp.util import mu_q_coupled, mu_q_coupled_two_states, get_state_basis


class TestMagneticBeams:
    def test_rf_field_properties(self):
        """Verify that RFField outputs correct magnetic field via unified field() method."""
        s = 2.0
        # Default polarization is [0, 0, 1] (Cartesian z-axis)
        beam = RFField(kvec=np.zeros(3), pol=[0.0, 0.0, 1.0], s=s, delta=0.0)

        r = np.array([0.1, 0.2, 0.3])
        t = 0.5
        # Field amplitude is B_0 = sqrt(2 * s) = sqrt(4) = 2.0
        B = beam.field(r, t)
        assert not np.allclose(B, np.zeros(3))
        np.testing.assert_allclose(np.linalg.norm(B), 2.0)

        # Gradient
        delB = beam.field_gradient(r, t)
        # With k=0 and constant s, spatial gradient is 0
        np.testing.assert_allclose(delB, np.zeros((3, 3)))

    def test_polarization_enum_disallowed_for_zero_kvec(self):
        """PolarizationEnum is only defined relative to k-vector; disallow for k=0 or RFField."""
        # LaserBeam with k=0 and PolarizationEnum must raise ValueError
        with pytest.raises(ValueError, match="PolarizationEnum is defined relative to the k-vector"):
            LaserBeam(kvec=np.zeros(3), pol=PolarizationEnum.SIGMA_PLUS, s=1.0)

        # RFField must disallow PolarizationEnum with TypeError
        with pytest.raises(TypeError, match="PolarizationEnum is only defined relative to a laser beam's k-vector"):
            RFField(pol=PolarizationEnum.SIGMA_PLUS)

    def test_scalar_polarizations_disallowed(self):
        """Scalar polarizations (+1, -1, 0) are legacy idioms and must raise TypeError."""
        for scalar_pol in [1, -1, 0, 1.0, -1.0]:
            with pytest.raises(TypeError, match="Scalar polarizations"):
                LaserBeam(kvec=np.array([0, 0, 1]), pol=scalar_pol, s=1.0)
            with pytest.raises(TypeError, match="Scalar polarizations"):
                RFField(pol=scalar_pol)

    def test_zero_kvec_rf_field(self):
        """RF near field with k=0 and Cartesian z-polarization."""
        # pol=[0, 0, 1] in Cartesian corresponds to q=0 (index 1) in spherical basis
        beam = RFField(kvec=np.zeros(3), pol=[0.0, 0.0, 1.0], s=1.0, delta=0.0)
        B_sph = beam.field(np.array([10.0, -5.0, 3.0]), t=0.0)
        np.testing.assert_allclose(B_sph[0], 0.0)  # q = -1
        np.testing.assert_allclose(B_sph[1], np.sqrt(2.0))  # q = 0
        np.testing.assert_allclose(B_sph[2], 0.0)  # q = +1

    def test_laser_beams_container(self):
        """Verify LaserBeams container correctly computes total_field."""
        mag_beam = RFField(kvec=np.zeros(3), pol=[0.0, 0.0, 1.0], s=1.0, delta=0.0)
        beams = LaserBeams([mag_beam])

        r = np.array([0.0, 0.0, 0.0])
        t = 0.0
        np.testing.assert_allclose(beams.total_field(r, t), np.array([0.0, np.sqrt(2.0), 0.0]))

    def test_rf_field_b0_amplitude(self):
        """Verify RFField directly supports B0 amplitude and maintains consistency with s."""
        b0_val = 3.0
        rf = RFField(B0=b0_val, pol=[0.0, 0.0, 1.0])
        assert rf.B0() == pytest.approx(3.0)
        assert rf.intensity() == pytest.approx(4.5)  # s = B0^2 / 2
        np.testing.assert_allclose(rf.field(np.zeros(3), 0.0), np.array([0.0, 3.0, 0.0]))

    def test_rf_field_polarizations(self):
        """Verify vector polarizations for RF fields."""
        # Cartesian z-coil [0, 0, 1] -> spherical q=0 (index 1)
        rf_z = RFField(B0=2.0, pol=[0.0, 0.0, 1.0])
        np.testing.assert_allclose(rf_z.field(np.zeros(3), 0.0), [0.0, 2.0, 0.0])

        # Spherical sigma+ basis vector [0, 0, 1] -> q = +1 (index 2)
        rf_sp = RFField(B0=2.0, pol=[0.0, 0.0, 1.0], pol_coord=CoordBasis.SPHERICAL if 'CoordBasis' in dir() else 1)

        # Cartesian linear x-coil: [1, 0, 0] -> spherical [1/sqrt(2), 0, -1/sqrt(2)]
        rf_cart = RFField(B0=2.0, pol=[1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            rf_cart.field(np.zeros(3), 0.0),
            np.array([2.0 / np.sqrt(2.0), 0.0, -2.0 / np.sqrt(2.0)])
        )


class TestTwoStateMuQ:
    def test_wigner_eckart_selection_rules(self):
        """Verify selection rules Delta m_F = q for inter-hyperfine mu_q."""
        # Rb87 5S1/2: J = 1/2, I = 3/2 -> F = 1 and F = 2
        # gJ ~ 2.0023, gI ~ -0.000995
        frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        state_g = frame.fine_states["g"]

        basis1 = get_state_basis(state_g, [1.0])  # F=1: mF = -1, 0, 1 (length 3)
        basis2 = get_state_basis(state_g, [2.0])  # F=2: mF = -2, -1, 0, 1, 2 (length 5)

        mu_q = mu_q_coupled_two_states(basis1, basis2, state_g.gJ, frame.gI, state_g.J, frame.I)
        assert mu_q.shape == (3, 3, 5)

        Fs1, mFs1 = basis1
        Fs2, mFs2 = basis2

        for q_idx, q in enumerate([-1.0, 0.0, 1.0]):
            block = mu_q[q_idx]
            for i1, mf1 in enumerate(mFs1):
                for i2, mf2 in enumerate(mFs2):
                    elem = block[i1, i2]
                    # Selection rule: mf2 = mf1 - q
                    if np.abs(mf2 - (mf1 - q)) > 1e-6:
                        assert np.abs(elem) < 1e-12, (
                            f"Non-zero element {elem} violates Delta m = {mf1 - mf2} != q={q}"
                        )

    def test_mu_q_two_states_hermiticity_symmetry(self):
        """Verify relation <F2, m2| mu_q |F1, m1> = (-1)^q <F1, m1| mu_{-q} |F2, m2>*."""
        frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        state_g = frame.fine_states["g"]
        basis1 = get_state_basis(state_g, [1.0])
        basis2 = get_state_basis(state_g, [2.0])

        mu_12 = mu_q_coupled_two_states(basis1, basis2, state_g.gJ, frame.gI, state_g.J, frame.I)
        mu_21 = mu_q_coupled_two_states(basis2, basis1, state_g.gJ, frame.gI, state_g.J, frame.I)

        # q = -1 (idx 0), q = 0 (idx 1), q = +1 (idx 2)
        # q = -1: mu_12[0] relates to mu_21[2] with (-1)^(-1) = -1
        np.testing.assert_allclose(mu_12[0], -np.conjugate(mu_21[2].T), atol=1e-12)
        # q = 0: mu_12[1] relates to mu_21[1] with (-1)^0 = +1
        np.testing.assert_allclose(mu_12[1], np.conjugate(mu_21[1].T), atol=1e-12)
        # q = +1: mu_12[2] relates to mu_21[0] with (-1)^1 = -1
        np.testing.assert_allclose(mu_12[2], -np.conjugate(mu_21[0].T), atol=1e-12)

    def test_single_basis_delegation(self):
        """Verify mu_q_coupled(basis, ...) delegates to mu_q_coupled_two_states(basis, basis, ...)."""
        frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        state_g = frame.fine_states["g"]
        basis = get_state_basis(state_g, [1.0, 2.0])

        res_single = mu_q_coupled(basis, state_g.gJ, frame.gI, state_g.J, frame.I)
        res_two = mu_q_coupled_two_states(basis, basis, state_g.gJ, frame.gI, state_g.J, frame.I)
        np.testing.assert_allclose(res_single, res_two, atol=1e-15)


class TestOBEMagneticTransitions:
    def test_two_level_resonant_rabi_oscillations(self):
        """Verify analytical Rabi oscillations rho_11(t) = sin^2(Omega * t / 2) with Omega = sqrt(s / 2)."""
        ham = Hamiltonian()
        ham.add_H_0_block('0', np.zeros((1, 1)))
        ham.add_H_0_block('1', np.zeros((1, 1)))

        mu_q = np.zeros((3, 1, 1), dtype=complex)
        mu_q[1, 0, 0] = 1.0  # pi polarization (q = 0)
        ham.add_mu_q_block('0', '0', np.zeros((3, 1, 1)))
        ham.add_mu_q_block('1', '1', np.zeros((3, 1, 1)))
        ham.add_mu_q_block('0', '1', mu_q, muB=1.0)

        s = 1.0
        beam = RFField(kvec=np.zeros(3), pol=[0.0, 0.0, 1.0], s=s, delta=0.0)
        beams = {TransitionKey('0', '1'): LaserBeams([beam])}
        mag = MagField(np.zeros(3))

        for transform_re_im in (True, False):
            obe = OBE(beams, mag, ham, transform_into_re_im=transform_re_im)
            obe.set_initial_rho(np.array([[1.0, 0.0], [0.0, 0.0]]))

            t_eval = np.linspace(0.0, 15.0, 301)
            sol = obe.evolve_density([0.0, 15.0], t_eval=t_eval, rtol=1e-10, atol=1e-10)

            Omega = np.sqrt(s / 2.0)
            analytical_rho11 = np.sin(Omega * t_eval / 2.0) ** 2
            analytical_rho00 = np.cos(Omega * t_eval / 2.0) ** 2

            np.testing.assert_allclose(np.real(sol.rho[1, 1, :]), analytical_rho11, atol=1e-6)
            np.testing.assert_allclose(np.real(sol.rho[0, 0, :]), analytical_rho00, atol=1e-6)
            # No spontaneous decay: Trace must remain 1.0 throughout
            traces = np.real(sol.rho[0, 0, :] + sol.rho[1, 1, :])
            np.testing.assert_allclose(traces, np.ones_like(traces), atol=1e-8)

    def test_two_level_detuned_rabi_oscillations(self):
        """Verify analytical detuned Rabi oscillations with generalized Rabi frequency."""
        ham = Hamiltonian()
        ham.add_H_0_block('0', np.zeros((1, 1)))
        ham.add_H_0_block('1', np.zeros((1, 1)))

        mu_q = np.zeros((3, 1, 1), dtype=complex)
        mu_q[1, 0, 0] = 1.0
        ham.add_mu_q_block('0', '0', np.zeros((3, 1, 1)))
        ham.add_mu_q_block('1', '1', np.zeros((3, 1, 1)))
        ham.add_mu_q_block('0', '1', mu_q, muB=1.0)

        s = 2.0
        delta = 0.75
        beam = RFField(kvec=np.zeros(3), pol=[0.0, 0.0, 1.0], s=s, delta=delta)
        beams = {TransitionKey('0', '1'): LaserBeams([beam])}
        mag = MagField(np.zeros(3))

        obe = OBE(beams, mag, ham, transform_into_re_im=True)
        obe.set_initial_rho(np.array([[1.0, 0.0], [0.0, 0.0]]))

        t_eval = np.linspace(0.0, 15.0, 301)
        sol = obe.evolve_density([0.0, 15.0], t_eval=t_eval, rtol=1e-10, atol=1e-10)

        Omega = np.sqrt(s / 2.0)
        Omega_prime = np.sqrt(Omega ** 2 + delta ** 2)
        analytical_rho11 = (Omega ** 2 / Omega_prime ** 2) * np.sin(Omega_prime * t_eval / 2.0) ** 2

        np.testing.assert_allclose(np.real(sol.rho[1, 1, :]), analytical_rho11, atol=1e-6)

    def test_full_obe_ev_matches_drhodt(self):
        """Verify full_OBE_ev @ rho matches __drhodt in transformed real/imaginary basis."""
        ham = Hamiltonian()
        ham.add_H_0_block('0', np.zeros((1, 1)))
        ham.add_H_0_block('1', np.zeros((1, 1)))

        mu_q = np.zeros((3, 1, 1), dtype=complex)
        mu_q[1, 0, 0] = 1.0
        ham.add_mu_q_block('0', '0', np.zeros((3, 1, 1)))
        ham.add_mu_q_block('1', '1', np.zeros((3, 1, 1)))
        ham.add_mu_q_block('0', '1', mu_q, muB=1.0)

        beam = RFField(kvec=np.zeros(3), pol=[0.0, 0.0, 1.0], s=1.0, delta=0.2)
        beams = {TransitionKey('0', '1'): LaserBeams([beam])}
        mag = MagField(np.zeros(3))

        obe = OBE(beams, mag, ham, transform_into_re_im=True)
        r = np.zeros(3)
        t = 1.23

        ev = obe.full_OBE_ev(r, t)
        # Random real state vector in transformed basis
        rng = np.random.default_rng(42)
        rho = rng.standard_normal(ev.shape[1])
        dr_from_ev = ev @ rho
        dr_from_func = obe._OBE__drhodt(r, t, rho)
        np.testing.assert_allclose(dr_from_ev, dr_from_func, atol=1e-12)


class TestDataFrameMagneticLasers:
    def test_add_magnetic_transition_and_laser(self):
        """Test adding magnetic transition and magnetic laser via DataFrame API."""
        frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        # Register magnetic transition between F=1 and F=2
        trans_key = frame.add_magnetic_transition('g', 1, 2, muB=1.0)
        assert isinstance(trans_key, HFTransitionKey)
        assert trans_key in frame.magnetic_transitions
        assert trans_key in frame.magnetic_lasers

        # Add magnetic laser
        frame.add_magnetic_laser('g', 1, 2, delta=0.0, kvec=np.zeros(3),
                                 pol=[0.0, 0.0, 1.0], intensity=50.0)
        laser_group = frame.magnetic_lasers[trans_key]
        assert len(laser_group.freq_groups) == 1

    def test_add_laser_rejects_same_fine_state(self):
        """Verify frame.add_laser(g, g, 1, 2, ...) raises ValueError."""
        frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        with pytest.raises(ValueError, match="cannot have identical labels"):
            frame.add_laser('g', 'g', 1, 2, delta=0.0, kvec=np.zeros(3),
                            pol=[0.0, 0.0, 1.0], intensity=1.0)

    def test_hyperfine_block_mode_activation(self):
        """Verify presence of magnetic transition forces Hyperfine Block Mode."""
        frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        assert frame._has_custom_hyperfine_coupling() is False

        frame.add_magnetic_laser('g', 1, 2, delta=0.0, kvec=np.zeros(3),
                                 pol=[0.0, 0.0, 1.0], intensity=1.0)
        assert frame._has_custom_hyperfine_coupling() is True

        ham = frame._hamiltonian()
        # Must have partitioned blocks for hyperfine manifolds
        assert 'g_F1' in ham.state_labels
        assert 'g_F2' in ham.state_labels

        lasers = frame._lasers()
        assert TransitionKey('g_F1', 'g_F2') in lasers
        beam = lasers[TransitionKey('g_F1', 'g_F2')].beam_vector[0]
        assert isinstance(beam, RFField)

    def test_magnetic_saturation_intensity_formula(self):
        """Verify I_sat,mag = (c * (h * gamma_ref)^2) / (mu_0 * mu_B^2)."""
        frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        i_sat_mag = frame._saturation_intensity_mag()

        ref_gamma, _ = frame._principal_gamma_and_energy()
        mu_B = physical_constants['Bohr magneton'][0]
        expected = (c * (h * ref_gamma) ** 2) / (mu_0 * mu_B ** 2)
        np.testing.assert_allclose(i_sat_mag, expected, rtol=1e-12)
        # For Rb87 (gamma ~ 6.066 MHz), I_sat,mag ~ 4.48e7 W/m^2
        assert 4.0e7 < i_sat_mag < 5.0e7

    def test_end_to_end_rb87_clock_transition(self):
        """End-to-end multi-level simulation of Rb87 clock transition (F=1, mF=0 <-> F=2, mF=0)."""
        frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        frame.remove_state('e1')
        frame.remove_state('e2')
        # Driven on resonance with pi polarization: delta m = 0 selection rule
        frame.add_magnetic_laser('g', 1, 2, delta=0.0, kvec=np.zeros(3),
                                 pol=[0.0, 0.0, 1.0], intensity=1.0)

        obe = frame.obe()
        # State order in g_F1: mF = -1, 0, 1 (indices 0, 1, 2)
        # State order in g_F2: mF = -2, -1, 0, 1, 2 (indices 3, 4, 5, 6, 7)
        # Initial state: |F=1, mF=0> -> index 1
        rho0 = np.zeros((8, 8), dtype=complex)
        rho0[1, 1] = 1.0
        obe.set_initial_rho(rho0)

        t_eval = np.linspace(0.0, 2.0, 21)
        sol = obe.evolve_density([0.0, 2.0], t_eval=t_eval, rtol=1e-8, atol=1e-10)

        # Pi polarization strictly couples mF=0 to mF=0:
        # Initial mF=0 in F=1:
        assert np.isclose(np.real(sol.rho[1, 1, 0]), 1.0, atol=1e-8)
        # Population transfers to mF=0 in F=2 (index 5):
        assert np.real(sol.rho[5, 5, -1]) > 0.4
        # Sum of populations in mF=0 manifolds must equal total population (1.0)
        # i.e., strict zero leakage to mF != 0 levels:
        pop_m0 = np.real(sol.rho[1, 1, :] + sol.rho[5, 5, :])
        np.testing.assert_allclose(pop_m0, np.ones_like(pop_m0), atol=1e-8)
        # Total trace is strictly 1.0 across all time steps:
        total_traces = np.real([np.trace(sol.rho[:, :, ti]) for ti in range(len(t_eval))])
        np.testing.assert_allclose(total_traces, np.ones_like(total_traces), atol=1e-8)

    def test_end_to_end_optical_and_rf_combined(self):
        """End-to-end simulation combining optical pumping (g->e1) and RF driving (F=1<->F=2)."""
        frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
        frame.remove_state('e2')
        # Add RF field driving g F=1 <-> F=2
        frame.add_magnetic_laser('g', 1, 2, delta=0.0, kvec=np.zeros(3),
                                 pol=[0.0, 0.0, 1.0], intensity=1.0)
        # Add optical laser pumping g F=2 -> e1 F=2
        frame.add_laser('g', 'e1', 2, 2, delta=0.0, kvec=[0, 0, 1],
                        pol=[1.0, 0.0, 0.0], intensity=1.0)

        obe = frame.obe()
        # Initial state purely in g_F1 (index 0)
        rho0 = np.zeros((obe.hamiltonian.n, obe.hamiltonian.n), dtype=complex)
        rho0[0, 0] = 1.0
        obe.set_initial_rho(rho0)

        t_eval = np.linspace(0.0, 2.0, 21)
        sol = obe.evolve_density([0.0, 2.0], t_eval=t_eval, rtol=1e-8, atol=1e-10)

        # Strict trace preservation to 1e-8:
        total_traces = np.real([np.trace(sol.rho[:, :, ti]) for ti in range(len(t_eval))])
        np.testing.assert_allclose(total_traces, np.ones_like(total_traces), atol=1e-8)

        # Confirm RF ground-state transfer to g_F2 (indices 3..7):
        g2_pop = np.real(np.sum([sol.rho[i, i, -1] for i in range(3, 8)]))
        assert g2_pop > 0.3  # ~33% transferred at t=2.0

        # Confirm optical excited state population (indices 8..15) builds up matching physics:
        excited_pop = np.real(np.sum([sol.rho[i, i, -1] for i in range(8, obe.hamiltonian.n)]))
        assert 0.002 < excited_pop < 0.004



class TestKeyTypingAndTransitionValidation:
    def test_governingeq_rejects_string_keys(self):
        """laserBeams dictionary keys must strictly be TransitionKey instances."""
        beam = LaserBeam(kvec=np.array([0, 0, 1]), pol=PolarizationEnum.SIGMA_PLUS, s=1.0)
        beams = LaserBeams([beam])
        ham = Hamiltonian()
        ham.add_H_0_block('g', np.array([[0.0]]))
        ham.add_H_0_block('e', np.array([[1.0]]))
        ham.add_d_q_block('g', 'e', np.ones((3, 1, 1)), gamma=1.0)
        mag = MagField(np.zeros(3))

        # String key 'g->e' must raise TypeError
        with pytest.raises(TypeError, match="laserBeams dictionary keys must be TransitionKey instances"):
            OBE({'g->e': beams}, mag, ham)

        # Integer key must raise TypeError
        with pytest.raises(TypeError, match="laserBeams dictionary keys must be TransitionKey instances"):
            OBE({42: beams}, mag, ham)

    def test_governingeq_rejects_untyped_laser_inputs(self):
        """LaserBeam or list must be rejected with TypeError (strictly LaserBeams or dict[TransitionKey, LaserBeams])."""
        beam = LaserBeam(kvec=np.array([0, 0, 1]), pol=PolarizationEnum.SIGMA_PLUS, s=1.0)
        ham = Hamiltonian()
        ham.add_H_0_block('g', np.array([[0.0]]))
        ham.add_H_0_block('e', np.array([[1.0]]))
        ham.add_d_q_block('g', 'e', np.ones((3, 1, 1)), gamma=1.0)
        mag = MagField(np.zeros(3))

        # Bare LaserBeam passed directly
        with pytest.raises(TypeError, match="laserBeams must be LaserBeams or dict"):
            OBE(beam, mag, ham)

        # Bare list passed directly
        with pytest.raises(TypeError, match="laserBeams must be LaserBeams or dict"):
            OBE([beam], mag, ham)

        # Dict with bare LaserBeam as value
        with pytest.raises(TypeError, match="Value for transition .* in laserBeams must be LaserBeams"):
            OBE({TransitionKey('g', 'e'): beam}, mag, ham)

        # Dict with list as value
        with pytest.raises(TypeError, match="Value for transition .* in laserBeams must be LaserBeams"):
            OBE({TransitionKey('g', 'e'): [beam]}, mag, ham)

    def test_governingeq_validates_transition_existence(self):
        """Laser addressing a non-existent transition key in Hamiltonian raises KeyError."""
        beam = LaserBeam(kvec=np.array([0, 0, 1]), pol=PolarizationEnum.SIGMA_PLUS, s=1.0)
        beams = LaserBeams([beam])
        ham = Hamiltonian()
        ham.add_H_0_block('g', np.array([[0.0]]))
        ham.add_H_0_block('e', np.array([[1.0]]))
        ham.add_d_q_block('g', 'e', np.ones((3, 1, 1)), gamma=1.0)
        mag = MagField(np.zeros(3))

        with pytest.raises(KeyError, match="does not exist in Hamiltonian"):
            OBE({TransitionKey('g', 'missing'): beams}, mag, ham)

    def test_governingeq_rejects_diagonal_keys(self):
        """Laser beams cannot address diagonal transitions within the same manifold."""
        beam = LaserBeam(kvec=np.array([0, 0, 1]), pol=PolarizationEnum.SIGMA_PLUS, s=1.0)
        beams = LaserBeams([beam])
        ham = Hamiltonian()
        ham.add_H_0_block('g', np.array([[0.0]]))
        ham.add_mu_q_block('g', 'g', np.zeros((3, 1, 1)))
        mag = MagField(np.zeros(3))

        with pytest.raises(ValueError, match="Laser beams cannot address diagonal transition"):
            OBE({TransitionKey('g', 'g'): beams}, mag, ham)

    def test_beam_type_and_parity_validation(self):
        """LaserBeam must have d_q, RFField must have mu_q, and cannot have both."""
        opt_beam = LaserBeam(kvec=np.array([0, 0, 1]), pol=PolarizationEnum.SIGMA_PLUS, s=1.0)
        rf_beam = RFField(B0=1.0, pol=[0.0, 0.0, 1.0])
        mag = MagField(np.zeros(3))

        # Optical transition Hamiltonian (only d_q)
        ham_opt = Hamiltonian()
        ham_opt.add_H_0_block('g', np.array([[0.0]]))
        ham_opt.add_H_0_block('e', np.array([[1.0]]))
        ham_opt.add_d_q_block('g', 'e', np.ones((3, 1, 1)), gamma=1.0)

        # Magnetic transition Hamiltonian (only off-diagonal mu_q)
        ham_mag = Hamiltonian()
        ham_mag.add_H_0_block('g', np.array([[0.0]]))
        ham_mag.add_H_0_block('e', np.array([[1.0]]))
        ham_mag.add_mu_q_block('g', 'e', np.ones((3, 1, 1)), muB=1.0)

        # 1. RFField on optical transition (has d_q, lacks mu_q) -> ValueError
        with pytest.raises(ValueError, match="RFField addresses transition .*, but no magnetic dipole"):
            OBE({TransitionKey('g', 'e'): LaserBeams([rf_beam])}, mag, ham_opt)

        # 2. LaserBeam on magnetic transition (has mu_q, lacks d_q) -> ValueError
        with pytest.raises(ValueError, match="LaserBeam addresses transition .*, but no electric dipole"):
            OBE({TransitionKey('g', 'e'): LaserBeams([opt_beam])}, mag, ham_mag)

        # 3. Transition with both d_q and mu_q -> ValueError (parity violation)
        ham_both = Hamiltonian()
        ham_both.add_H_0_block('g', np.array([[0.0]]))
        ham_both.add_H_0_block('e', np.array([[1.0]]))
        ham_both.add_d_q_block('g', 'e', np.ones((3, 1, 1)), gamma=1.0)
        ham_both.add_mu_q_block('g', 'e', np.ones((3, 1, 1)), muB=1.0)

        with pytest.raises(ValueError, match="violates parity conservation"):
            OBE({TransitionKey('g', 'e'): LaserBeams([opt_beam])}, mag, ham_both)

        with pytest.raises(ValueError, match="violates parity conservation"):
            OBE({TransitionKey('g', 'e'): LaserBeams([rf_beam])}, mag, ham_both)

        # 4. Correct pairings must succeed
        obe_opt = OBE({TransitionKey('g', 'e'): LaserBeams([opt_beam])}, mag, ham_opt)
        assert TransitionKey('g', 'e') in obe_opt.electric_lasers
        assert TransitionKey('g', 'e') in obe_opt.laserBeams
        obe_mag = OBE({TransitionKey('g', 'e'): LaserBeams([rf_beam])}, mag, ham_mag)
        assert TransitionKey('g', 'e') in obe_mag.magnetic_lasers
        assert TransitionKey('g', 'e') in obe_mag.laserBeams

    def test_return_full_h_strictly_requires_dict_of_transition_keys(self):
        """return_full_H must reject list/ndarray and require dict[TransitionKey, np.ndarray]."""
        ham = Hamiltonian()
        ham.add_H_0_block('g', np.array([[0.0]]))
        ham.add_H_0_block('e', np.array([[1.0]]))
        ham.add_d_q_block('g', 'e', np.ones((3, 1, 1)), gamma=1.0)
        ham.make_full_matrices()

        field_vec = np.zeros(3)

        # Eq as list or ndarray must raise TypeError
        with pytest.raises(TypeError, match="Eq must be a dict"):
            ham.return_full_H(Bq=field_vec, Eq=[field_vec])
        with pytest.raises(TypeError, match="Eq must be a dict"):
            ham.return_full_H(Bq=field_vec, Eq=field_vec)

        # Eq with string key must raise TypeError
        with pytest.raises(TypeError, match="Eq dictionary keys must be TransitionKey"):
            ham.return_full_H(Bq=field_vec, Eq={'g->e': field_vec})

        # B_rf as list or ndarray must raise TypeError
        with pytest.raises(TypeError, match="B_rf must be a dict"):
            ham.return_full_H(Bq=field_vec, B_rf=[field_vec])
        with pytest.raises(TypeError, match="B_rf must be a dict"):
            ham.return_full_H(Bq=field_vec, B_rf=field_vec)

        # B_rf with string key must raise TypeError
        with pytest.raises(TypeError, match="B_rf dictionary keys must be TransitionKey"):
            ham.return_full_H(Bq=field_vec, B_rf={'g->e': field_vec})

        # Proper dict[TransitionKey, np.ndarray] works as expected
        H = ham.return_full_H(Bq=field_vec, Eq={TransitionKey('g', 'e'): field_vec})
        assert H.shape == (2, 2)


class TestHamiltonianVectorBlockAndMuQRestructuring:
    def test_vector_block_add_dagger_to_other_in_place(self):
        """VectorBlock.add_dagger_to_other_in_place must place M^\\dagger at (j, i)."""
        M = np.array([
            [[1.0 + 2.0j, 3.0 - 1.0j]],
            [[0.5, -2.5j]],
            [[4.0j, 1.0]]
        ])  # shape (3, 1, 2)
        vb = VectorBlock('g', 'e', M)
        other = np.zeros((3, 4, 4), dtype='complex128')
        vb.add_dagger_to_other_in_place(other, 1, 2, scale=1.0)
        # Expected shape of M^\dagger is (3, 2, 1) placed at rows 2:4, cols 1:2
        for q in range(3):
            expected_block = np.conjugate(M[q].T)
            np.testing.assert_allclose(other[q, 2:4, 1:2], expected_block)
            other_copy = other.copy()
            other_copy[q, 2:4, 1:2] = 0.0
            assert np.all(other_copy[q] == 0.0)

    def test_full_block_is_diagonal_with_equal_manifold_dimensions(self):
        """FullBlock.is_diagonal must be False for off-diagonal blocks even when n == m."""
        ham = Hamiltonian()
        ham.add_H_0_block('g', np.zeros((3, 3)))
        ham.add_H_0_block('e', np.zeros((3, 3)))
        fb_diag_g = ham.blocks_by_keys[TransitionKey('g', 'g')]
        fb_diag_e = ham.blocks_by_keys[TransitionKey('e', 'e')]
        fb_off_diag = ham.blocks_by_keys[TransitionKey('g', 'e')]

        assert fb_diag_g.is_diagonal is True
        assert fb_diag_e.is_diagonal is True
        assert fb_off_diag.is_diagonal is False

    def test_hamiltonian_vector_block_storage(self):
        """d_q_bare and mu_bare must store VectorBlock objects, not full ndarrays."""
        ham = Hamiltonian()
        ham.add_H_0_block('g', np.array([[0.0]]))
        ham.add_H_0_block('e', np.array([[1.0]]))
        ham.add_mu_q_block('g', 'g', np.ones((3, 1, 1)), muB=1.5)
        ham.add_mu_q_block('e', 'e', np.ones((3, 1, 1)) * 2, muB=1.5)
        ham.add_mu_q_block('g', 'e', np.ones((3, 1, 1)) * 3, muB=1.5)
        ham.add_d_q_block('g', 'e', np.ones((3, 1, 1)), gamma=1.0)

        ham.make_full_matrices()

        # d_q_bare must contain DQBlock instance
        key_ge = TransitionKey('g', 'e')
        assert key_ge in ham.d_q_bare
        assert isinstance(ham.d_q_bare[key_ge], DQBlock)
        assert not isinstance(ham.d_q_bare[key_ge], np.ndarray)

        # mu_bare must contain MuQBlock instance for off-diagonal transition
        assert key_ge in ham.mu_bare
        assert isinstance(ham.mu_bare[key_ge], MuQBlock)
        assert not isinstance(ham.mu_bare[key_ge], np.ndarray)

        # Diagonal mu_q blocks must NOT be in mu_bare
        assert TransitionKey('g', 'g') not in ham.mu_bare
        assert TransitionKey('e', 'e') not in ham.mu_bare

        # ham.mu_q must contain only intra-manifold (diagonal) elements
        np.testing.assert_allclose(ham.mu_q[:, 0, 0], np.ones(3) * 1.5)
        np.testing.assert_allclose(ham.mu_q[:, 1, 1], np.ones(3) * 3.0)
        # Off-diagonal blocks in static mu_q must strictly be zero:
        np.testing.assert_allclose(ham.mu_q[:, 0, 1], np.zeros(3))
        np.testing.assert_allclose(ham.mu_q[:, 1, 0], np.zeros(3))

    def test_hamiltonian_return_full_h_numerical_hermiticity(self):
        """return_full_H must produce the exact Hermitian Hamiltonian matrix."""
        ham = Hamiltonian()
        ham.add_H_0_block('g', np.diag([0.0, 0.5]))
        ham.add_H_0_block('e', np.diag([10.0, 12.0]))
        # Magnetic intra-manifold
        mu_g = np.zeros((3, 2, 2), dtype='complex128')
        mu_g[1] = np.diag([1.0, -1.0])
        ham.add_mu_q_block('g', 'g', mu_g, muB=1.0)
        # Magnetic off-diagonal RF transition
        mu_ge = np.zeros((3, 2, 2), dtype='complex128')
        mu_ge[1, 0, 1] = 0.5 + 0.2j
        ham.add_mu_q_block('g', 'e', mu_ge, muB=1.0)

        # Optical transition
        dq_ge = np.zeros((3, 2, 2), dtype='complex128')
        dq_ge[2, 0, 0] = 1.0 - 0.5j
        ham.add_d_q_block('g', 'e', dq_ge, gamma=1.0)

        ham.make_full_matrices()

        Bq = np.array([0.0, 2.0, 0.0])  # static field along z
        Eq_vec = np.array([0.1, 0.0, 0.2])
        Brf_vec = np.array([0.0, 0.05, 0.0])

        key_ge = TransitionKey('g', 'e')
        H = ham.return_full_H(Bq=Bq, Eq={key_ge: Eq_vec}, B_rf={key_ge: Brf_vec})

        # H must be strictly Hermitian
        np.testing.assert_allclose(H, H.conj().T, atol=1e-14)

        # Check static Zeeman contribution on ground manifold (indices 0:2, 0:2)
        # H0 - mu_q * Bq:
        # Bq[1] = 2.0, mu_g[1] = diag([1.0, -1.0]) -> -mu_q*Bq = -[2.0, -2.0] = [-2.0, 2.0]
        # H0_g = diag([0.0, 0.5]) -> H[0,0] = -2.0, H[1,1] = 2.5
        assert np.isclose(H[0, 0], -2.0)
        assert np.isclose(H[1, 1], 2.5)

        # Check that off-diagonal blocks receive the driving optical and RF terms
        assert not np.isclose(H[0, 2], 0.0)
        assert not np.isclose(H[0, 3], 0.0)


class TestLaserPartitioningAndSolverDecoupling:
    def test_governingeq_partitions_electric_and_magnetic_lasers(self):
        """GoverningEq must partition beams into electric_lasers and magnetic_lasers."""
        ham = Hamiltonian()
        ham.add_H_0_block('g', np.array([[0.0]]))
        ham.add_H_0_block('e', np.array([[10.0]]))
        ham.add_H_0_block('m', np.array([[1.0]]))

        # Optical transition g->e
        ham.add_d_q_block('g', 'e', np.ones((3, 1, 1)), gamma=1.0)
        # Magnetic RF transition g->m
        ham.add_mu_q_block('g', 'm', np.ones((3, 1, 1)), muB=1.0)

        opt_beam = LaserBeam(kvec=np.array([0, 0, 1]), pol=[1.0, 0.0, 0.0], s=1.0)
        rf_beam = RFField(kvec=np.zeros(3), pol=[0.0, 0.0, 1.0], s=1.0)

        mag = MagField(np.zeros(3))
        key_opt = TransitionKey('g', 'e')
        key_rf = TransitionKey('g', 'm')

        obe = OBE({key_opt: LaserBeams([opt_beam]), key_rf: LaserBeams([rf_beam])}, mag, ham)

        # Check partitioning:
        assert key_opt in obe.electric_lasers
        assert key_opt not in obe.magnetic_lasers
        assert key_rf in obe.magnetic_lasers
        assert key_rf not in obe.electric_lasers

        # Check unified laserBeams property:
        assert key_opt in obe.laserBeams
        assert key_rf in obe.laserBeams
        assert len(obe.laserBeams) == 2

    def test_rateeq_rejects_magnetic_lasers(self):
        """RateEq must reject magnetic (RF) laser beams with an explanatory error."""
        ham = Hamiltonian()
        ham.add_H_0_block('g', np.array([[0.0]]))
        ham.add_H_0_block('m', np.array([[1.0]]))
        ham.add_mu_q_block('g', 'm', np.ones((3, 1, 1)), muB=1.0)

        rf_beam = RFField(kvec=np.zeros(3), pol=[0.0, 0.0, 1.0], s=1.0)
        mag = MagField(np.zeros(3))

        with pytest.raises(NotImplementedError, match="RateEq only models spontaneous-scattering"):
            RateEq({TransitionKey('g', 'm'): LaserBeams([rf_beam])}, mag, ham)

    def test_rateeq_accepts_electric_lasers(self):
        """RateEq must accept electric lasers and calculate evolution using electric_lasers."""
        ham = Hamiltonian()
        ham.add_H_0_block('g', np.array([[0.0]]))
        ham.add_H_0_block('e', np.array([[1.0]]))
        ham.add_d_q_block('g', 'e', np.ones((3, 1, 1)), gamma=1.0)

        opt_beam = LaserBeam(kvec=np.array([0, 0, 1]), pol=[1.0, 0.0, 0.0], s=1.0)
        mag = MagField(np.zeros(3))

        rateeq = RateEq({TransitionKey('g', 'e'): LaserBeams([opt_beam])}, mag, ham)
        assert TransitionKey('g', 'e') in rateeq.electric_lasers
        assert len(rateeq.magnetic_lasers) == 0

        # Pumping rates must evaluate without error
        rateeq.construct_evolution_matrix(np.zeros(3), np.zeros(3))
        assert TransitionKey('g', 'e') in rateeq.Rijl

    def test_static_magnetic_field_strictly_intra_manifold(self):
        """obe.ev_mat.magnetic must strictly couple intra-manifold states only under RWA."""
        ham = Hamiltonian()
        ham.add_H_0_block('g', np.array([[0.0, 0.0], [0.0, 0.0]]))
        ham.add_H_0_block('m', np.array([[1.0, 0.0], [0.0, 1.0]]))
        ham.add_H_0_block('e', np.array([[10.0, 0.0], [0.0, 10.0]]))

        # Intra-manifold mu_q on ground manifold
        mu_g = np.zeros((3, 2, 2))
        mu_g[1] = np.diag([1.0, -1.0])
        ham.add_mu_q_block('g', 'g', mu_g, muB=1.0)

        # Inter-manifold off-diagonal mu_q on RF transition g->m
        mu_gm = np.zeros((3, 2, 2))
        mu_gm[1] = np.ones((2, 2)) * 5.0
        ham.add_mu_q_block('g', 'm', mu_gm, muB=1.0)

        # Optical transition g->e
        ham.add_d_q_block('g', 'e', np.ones((3, 2, 2)), gamma=1.0)

        opt_beam = LaserBeam(kvec=np.array([0, 0, 1]), pol=[1.0, 0.0, 0.0], s=1.0)
        rf_beam = RFField(kvec=np.zeros(3), pol=[0.0, 0.0, 1.0], s=1.0)
        mag = MagField(np.zeros(3))

        obe = OBE({
            TransitionKey('g', 'e'): LaserBeams([opt_beam]),
            TransitionKey('g', 'm'): LaserBeams([rf_beam])
        }, mag, ham, transform_into_re_im=False)

        # obe.ev_mat.magnetic must match the commutator with static intra-manifold ham.mu_q
        In = np.eye(ham.n, dtype='complex128')
        for q in range(3):
            expected_mag_ev = 1j * (np.kron(obe.hamiltonian.mu_q[q].T, In) - np.kron(In, obe.hamiltonian.mu_q[q]))
            np.testing.assert_allclose(obe.ev_mat.magnetic[q], expected_mag_ev, atol=1e-12)

