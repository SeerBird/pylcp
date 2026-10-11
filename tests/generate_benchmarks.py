"""
Script to generate and save baseline upstream PyLCP reference benchmarks.

This script executes the reference simulations and function calls using upstream PyLCP
and saves the resulting arrays and execution times to compressed .npz files in tests/data/.
These baseline files allow the pytest suite to verify fork correctness and speedup
without re-running the sluggish upstream code on every test invocation.
"""

import sys
import os
import time
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import importlib
import tests.vendor.upstream_pylcp as upstream
upstream_atom = importlib.import_module("tests.vendor.upstream_pylcp.atom")


def generate_all_benchmarks(data_dir: str = "tests/data"):
    os.makedirs(data_dir, exist_ok=True)
    print("=" * 60)
    print("Generating Upstream PyLCP Reference Benchmarks")
    print("=" * 60)
    start_time = time.perf_counter()
    # region Atom Benchmarks
    print("\n[1/9] Generating Atom benchmarks...")
    up_state = upstream_atom.state(n=5, S=0.5, L=1, J=1.5, lam=780e-9, tau=26e-9, gJ=1.33)
    up_rb = upstream_atom.atom("87Rb")
    np.savez_compressed(
        os.path.join(data_dir, "atom_ref.npz"),
        state_energy=up_state.energy,
        state_gamma=up_state.gamma,
        state_gammaHz=up_state.gammaHz,
        state_gJ=up_state.gJ,
        rb_I=up_rb.I,
        rb_gI=up_rb.gI,
        rb_mass=up_rb.mass,
        rb_states_energy=np.array([s.energy for s in up_rb.state]),
        rb_states_gJ=np.array([s.gJ for s in up_rb.state]),
    )
    print("  -> Saved atom_ref.npz")
    # endregion
    # region Fields Benchmarks
    print("\n[2/9] Generating Fields benchmarks...")
    B0 = np.array([1.2, -0.5, 3.4])
    mag_const = upstream.constantMagneticField(B0)
    R_const = np.array([0.5, -1.0, 2.0])
    const_field = mag_const.Field(R_const, 0)
    const_mag = mag_const.FieldMag(R_const, 0)
    const_grad = mag_const.gradField(R_const, 0)

    grad = 15.0
    mag_quad = upstream.quadrupoleMagneticField(grad)
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, -2.0, 3.0]),
        np.array([-0.5, 0.5, -1.0])
    ]
    quad_fields = np.array([mag_quad.Field(r, 0) for r in positions])
    quad_grad_mags = np.array([mag_quad.gradFieldMag(r, 0) for r in positions])

    beams_mot = upstream.conventional3DMOTBeams(s=2.0, delta=-1.5)
    R0 = np.zeros(3)
    mot_deltas = np.array([b.delta(0) for b in beams_mot.beam_vector])
    mot_kvecs = np.array([b.kvec(R0, 0) for b in beams_mot.beam_vector])

    np.savez_compressed(
        os.path.join(data_dir, "fields_ref.npz"),
        const_field=const_field,
        const_mag=const_mag,
        const_grad=const_grad,
        quad_fields=quad_fields,
        quad_grad_mags=quad_grad_mags,
        mot_deltas=mot_deltas,
        mot_kvecs=mot_kvecs,
    )
    print("  -> Saved fields_ref.npz")
    # endregion
    # region Hamiltonians Benchmarks
    print("\n[3/9] Generating Hamiltonians benchmarks...")
    H0_sf, muq_sf = upstream.hamiltonians.singleF(F=1, gF=1.5)
    H0_hf, muq_hf = upstream.hamiltonians.hyperfine_coupled(
        J=0.5, I=1.5, gJ=2.0, gI=-0.000995, Ahfs=3.417e9
    )
    H0_fs, muq_fs = upstream.hamiltonians.fine_structure_uncoupled(
        L=1, S=0.5, I=1.5, xi=1e9, a_c=1e6, a_orb=1e5, a_dip=1e4,
        gL=1.0, gS=2.0, gI=-0.001
    )
    rng = np.random.RandomState(42)
    dq_raw = rng.rand(3, 8, 12)
    dq_norm = upstream.hamiltonians.dqij_norm(dq_raw)
    dq_two_hf = upstream.hamiltonians.dqij_two_hyperfine_manifolds(J=0.5, Jp=1.5, I=1.5)
    dq_bare_hf = upstream.hamiltonians.dqij_two_bare_hyperfine(F=1, Fp=2, normalize=True)

    H0_g, muq_g = upstream.hamiltonians.singleF(F=0, gF=1.0)
    H0_e, muq_e = upstream.hamiltonians.singleF(F=1, gF=1.0)
    dqij = upstream.hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    H_obj = upstream.hamiltonian(H0_g, H0_e, muq_g, muq_e, dqij, mass=87.0)

    np.savez_compressed(
        os.path.join(data_dir, "hamiltonians_ref.npz"),
        H0_sf=H0_sf,
        muq_sf=muq_sf,
        H0_hf=H0_hf,
        muq_hf=muq_hf,
        H0_fs=H0_fs,
        muq_fs=muq_fs,
        dq_raw=dq_raw,
        dq_norm=dq_norm,
        dq_two_hf=dq_two_hf,
        dq_bare_hf=dq_bare_hf,
        ham_ns=np.array(H_obj.ns),
        ham_n=H_obj.n,
        ham_blocks_len=len(H_obj.blocks),
    )
    print("  -> Saved hamiltonians_ref.npz")
    # endregion
    # region OBE Static Evaluation & Force Profile Benchmarks
    print("\n[4/9] Generating OBE static reference...")
    beams_obe = upstream.conventional3DMOTBeams(s=1.0, delta=-1.0)
    mag_const0 = upstream.constantMagneticField(np.array([0.0, 0.0, 0.0]))
    eq_obe = upstream.obe(beams_obe, mag_const0, H_obj, transform_into_re_im=False)
    r_obe = np.array([0.1, -0.2, 0.3])
    ev_obe = eq_obe.full_OBE_ev(r_obe, 0.0)

    grid = np.linspace(-0.5, 0.5, 2)
    R_grid = np.array([grid, np.zeros(2), np.zeros(2)])
    V_grid = np.array([np.zeros(2), np.zeros(2), np.zeros(2)])
    eq_obe.generate_force_profile(R_grid, V_grid, name="prof")
    obe_prof_F = eq_obe.profile["prof"].F

    np.savez_compressed(
        os.path.join(data_dir, "obe_ref.npz"),
        ev_mat=ev_obe,
        force_prof_F=obe_prof_F,
    )
    print("  -> Saved obe_ref.npz")
    # endregion
    # region RateEq Equilibrium & Force Profile Benchmarks
    print("\n[5/9] Generating RateEq static reference...")
    mag_quad15 = upstream.quadrupoleMagneticField(15.0)
    eq_rate = upstream.rateeq(beams_obe, mag_quad15, H_obj)
    neq_0 = eq_rate.equilibrium_populations(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), t=0)
    neq_1 = eq_rate.equilibrium_populations(np.array([0.1, -0.2, 0.3]), np.array([1.0, 0.0, -0.5]), t=0)

    eq_rate.generate_force_profile(R_grid, V_grid, name="prof")
    rate_prof_F = eq_rate.profile["prof"].F

    np.savez_compressed(
        os.path.join(data_dir, "rateeq_ref.npz"),
        neq_0=neq_0,
        neq_1=neq_1,
        force_prof_F=rate_prof_F,
    )
    print("  -> Saved rateeq_ref.npz")
    # endregion
    # region RateEq Populations (Optical Pumping / Relaxation)
    print("\n[6/9] Running Upstream RateEq evolve_populations...")
    t0 = time.perf_counter()
    eq_pop = upstream.rateeq(beams_obe, mag_quad15, H_obj)
    r0_pop = np.array([0.1, -0.2, 0.3])
    v0_pop = np.array([0.5, 0.0, -0.2])
    eq_pop.set_initial_position_and_velocity(r0_pop, v0_pop)
    N0_pop = np.array([1.0, 0.0, 0.0, 0.0])
    eq_pop.set_initial_pop(N0_pop)

    t_eval_pop = np.linspace(0, 5.0, 26)
    sol_pop = eq_pop.evolve_populations([0, 5.0], t_eval=t_eval_pop)
    t_pop = time.perf_counter() - t0

    np.savez_compressed(
        os.path.join(data_dir, "rateeq_populations_ref.npz"),
        t=sol_pop.t,
        y=sol_pop.y,
        r0=r0_pop,
        v0=v0_pop,
        N0=N0_pop,
        upstream_time=t_pop,
    )
    print(f"  -> Saved rateeq_populations_ref.npz (Upstream time: {t_pop:.4f}s)")
    # endregion
    # region RateEq Motion (3D MOT Trajectory)
    print("\n[7/9] Running Upstream RateEq evolve_motion...")
    t0 = time.perf_counter()
    eq_mot = upstream.rateeq(beams_obe, mag_quad15, H_obj)
    r0_mot = np.array([0.2, 0.1, -0.1])
    v0_mot = np.array([1.0, -0.5, 0.2])
    eq_mot.set_initial_position_and_velocity(r0_mot, v0_mot)
    eq_mot.set_initial_pop_from_equilibrium()

    t_eval_mot = np.linspace(0, 2.0, 21)
    sol_mot = eq_mot.evolve_motion([0, 2.0], t_eval=t_eval_mot)
    t_mot = time.perf_counter() - t0

    np.savez_compressed(
        os.path.join(data_dir, "rateeq_motion_ref.npz"),
        t=sol_mot.t,
        r=sol_mot.r,
        v=sol_mot.v,
        N=sol_mot.N,
        r0=r0_mot,
        v0=v0_mot,
        upstream_time=t_mot,
    )
    print(f"  -> Saved rateeq_motion_ref.npz (Upstream time: {t_mot:.4f}s)")
    # endregion
    # region OBE Density (Rabi Oscillations & Spontaneous Damping)
    print("\n[8/9] Running Upstream OBE evolve_density...")
    t0 = time.perf_counter()
    beam_rabi = upstream.laserBeam(kvec=np.array([0, 0, 1]), pol=np.array([0, 0, 1]), s=2.0, delta=0.0)
    beams_rabi = upstream.laserBeams([beam_rabi])

    obe_dens = upstream.obe(beams_rabi, mag_const0, H_obj, transform_into_re_im=True)
    obe_dens.set_initial_rho_equally()

    t_eval_dens = np.linspace(0, 5.0, 51)
    sol_dens = obe_dens.evolve_density([0, 5.0], t_eval=t_eval_dens)
    t_dens = time.perf_counter() - t0

    np.savez_compressed(
        os.path.join(data_dir, "obe_density_ref.npz"),
        t=sol_dens.t,
        rho=sol_dens.rho,
        upstream_time=t_dens,
    )
    print(f"  -> Saved obe_density_ref.npz (Upstream time: {t_dens:.4f}s)")
    # endregion
    # region OBE Motion (Coupled Optical Trajectory)
    print("\n[9/9] Running Upstream OBE evolve_motion...")
    t0 = time.perf_counter()
    beam_obe_mot = upstream.laserBeam(kvec=np.array([0, 0, 1]), pol=1, s=1.0, delta=-1.0)
    beams_obe_mot = upstream.laserBeams([beam_obe_mot])

    obe_mot = upstream.obe(beams_obe_mot, mag_const0, H_obj, transform_into_re_im=True)
    r0_obe = np.array([0.0, 0.0, 0.0])
    v0_obe = np.array([0.0, 0.0, 0.5])
    obe_mot.set_initial_position_and_velocity(r0_obe, v0_obe)
    obe_mot.set_initial_rho_equally()

    t_eval_obe_mot = np.linspace(0, 0.05, 6)
    sol_obe_mot = obe_mot.evolve_motion([0, 0.05], t_eval=t_eval_obe_mot)
    t_obe_mot = time.perf_counter() - t0

    np.savez_compressed(
        os.path.join(data_dir, "obe_motion_ref.npz"),
        t=sol_obe_mot.t,
        r=sol_obe_mot.r,
        v=sol_obe_mot.v,
        rho=sol_obe_mot.rho,
        r0=r0_obe,
        v0=v0_obe,
        upstream_time=t_obe_mot,
    )
    print(f"  -> Saved obe_motion_ref.npz (Upstream time: {t_obe_mot:.4f}s)")
    # endregion
    print("\n" + "=" * 60)
    print(f"All upstream benchmark reference files generated successfully in "+
          f"{time.perf_counter()-start_time} s!")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_benchmarks()
