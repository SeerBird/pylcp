"""
Script to generate and save baseline upstream PyLCP reference benchmarks.

This script executes the reference simulations and function calls using upstream PyLCP
and saves the resulting arrays and execution times to compressed .npz files in tests/data/.
These baseline files allow the pytest suite to verify fork correctness and speedup
without re-running the sluggish upstream code on every test invocation.

All simulation configurations (states, fields, lasers, grids, time spans) are imported
from `tests.test_configs` so that tests and benchmark generation stay in lockstep.
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

from tests.test_configs import (
    ATOM_STATE_CONFIG, ATOM_RB_SPECIES,
    CONST_B_VECTOR, CONST_B_POS, QUAD_B_GRADIENT, QUAD_B_POSITIONS, MOT_BEAMS_CONFIG, ZERO_B_VECTOR,
    SINGLE_F_CONFIG, HYPERFINE_COUPLED_CONFIG, FINE_STRUCTURE_CONFIG,
    DQ_TWO_HYPERFINE_CONFIG, DQ_BARE_HYPERFINE_CONFIG,
    TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED, TWO_LEVEL_DQIJ, TWO_LEVEL_MASS,
    OBE_MOT_BEAMS_CONFIG, OBE_EVAL_R, OBE_EVAL_T,
    FORCE_GRID_R, FORCE_GRID_V,
    OBE_RABI_BEAM_CONFIG, OBE_DENSITY_TIME,
    OBE_MOTION_BEAM_CONFIG, OBE_MOTION_R0, OBE_MOTION_V0, OBE_MOTION_TIME,
    RATEEQ_EQUILIBRIUM_TEST_POINTS,
    RATEEQ_POP_R0, RATEEQ_POP_V0, RATEEQ_POP_N0, RATEEQ_POP_TIME,
    RATEEQ_MOTION_R0, RATEEQ_MOTION_V0, RATEEQ_MOTION_TIME,
)


def generate_all_benchmarks(data_dir: str = "tests/data"):
    os.makedirs(data_dir, exist_ok=True)
    print("=" * 60)
    print("Generating Upstream PyLCP Reference Benchmarks")
    print("=" * 60)
    start_time = time.perf_counter()

    # region Atom Benchmarks
    print("\n[1/9] Generating Atom benchmarks...")
    up_state = upstream_atom.state(**ATOM_STATE_CONFIG)
    up_rb = upstream_atom.atom(ATOM_RB_SPECIES)
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
    mag_const = upstream.constantMagneticField(CONST_B_VECTOR)
    const_field = mag_const.Field(CONST_B_POS, 0)
    const_mag = mag_const.FieldMag(CONST_B_POS, 0)
    const_grad = mag_const.gradField(CONST_B_POS, 0)

    mag_quad = upstream.quadrupoleMagneticField(QUAD_B_GRADIENT)
    quad_fields = np.array([mag_quad.Field(r, 0) for r in QUAD_B_POSITIONS])
    quad_grad_mags = np.array([mag_quad.gradFieldMag(r, 0) for r in QUAD_B_POSITIONS])

    beams_mot = upstream.conventional3DMOTBeams(**MOT_BEAMS_CONFIG)
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
    H0_sf, muq_sf = upstream.hamiltonians.singleF(**SINGLE_F_CONFIG)
    H0_hf, muq_hf = upstream.hamiltonians.hyperfine_coupled(**HYPERFINE_COUPLED_CONFIG)
    H0_fs, muq_fs = upstream.hamiltonians.fine_structure_uncoupled(**FINE_STRUCTURE_CONFIG)

    rng = np.random.RandomState(42)
    dq_raw = rng.rand(3, 8, 12)
    dq_norm = upstream.hamiltonians.dqij_norm(dq_raw)
    dq_two_hf = upstream.hamiltonians.dqij_two_hyperfine_manifolds(**DQ_TWO_HYPERFINE_CONFIG)
    dq_bare_hf = upstream.hamiltonians.dqij_two_bare_hyperfine(**DQ_BARE_HYPERFINE_CONFIG)

    H0_g, muq_g = upstream.hamiltonians.singleF(**TWO_LEVEL_GROUND)
    H0_e, muq_e = upstream.hamiltonians.singleF(**TWO_LEVEL_EXCITED)
    dqij = upstream.hamiltonians.dqij_two_bare_hyperfine(**TWO_LEVEL_DQIJ)
    H_obj = upstream.hamiltonian(H0_g, H0_e, muq_g, muq_e, dqij, mass=TWO_LEVEL_MASS)

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
    beams_obe = upstream.conventional3DMOTBeams(**OBE_MOT_BEAMS_CONFIG)
    mag_const0 = upstream.constantMagneticField(ZERO_B_VECTOR)
    eq_obe = upstream.obe(beams_obe, mag_const0, H_obj, transform_into_re_im=False)
    ev_obe = eq_obe.full_OBE_ev(OBE_EVAL_R, OBE_EVAL_T)

    eq_obe.generate_force_profile(FORCE_GRID_R, FORCE_GRID_V, name="prof")
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
    mag_quad15 = upstream.quadrupoleMagneticField(QUAD_B_GRADIENT)
    eq_rate = upstream.rateeq(beams_obe, mag_quad15, H_obj)
    neq_0 = eq_rate.equilibrium_populations(RATEEQ_EQUILIBRIUM_TEST_POINTS[0][0], RATEEQ_EQUILIBRIUM_TEST_POINTS[0][1], t=0)
    neq_1 = eq_rate.equilibrium_populations(RATEEQ_EQUILIBRIUM_TEST_POINTS[1][0], RATEEQ_EQUILIBRIUM_TEST_POINTS[1][1], t=0)

    eq_rate.generate_force_profile(FORCE_GRID_R, FORCE_GRID_V, name="prof")
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
    eq_pop.set_initial_position_and_velocity(RATEEQ_POP_R0, RATEEQ_POP_V0)
    eq_pop.set_initial_pop(RATEEQ_POP_N0)

    sol_pop = eq_pop.evolve_populations(RATEEQ_POP_TIME["t_span"], t_eval=RATEEQ_POP_TIME["t_eval"])
    t_pop = time.perf_counter() - t0

    np.savez_compressed(
        os.path.join(data_dir, "rateeq_populations_ref.npz"),
        t=sol_pop.t,
        y=sol_pop.y,
        r0=RATEEQ_POP_R0,
        v0=RATEEQ_POP_V0,
        N0=RATEEQ_POP_N0,
        upstream_time=t_pop,
    )
    print(f"  -> Saved rateeq_populations_ref.npz (Upstream time: {t_pop:.4f}s)")
    # endregion

    # region RateEq Motion (3D MOT Trajectory)
    print("\n[7/9] Running Upstream RateEq evolve_motion...")
    t0 = time.perf_counter()
    eq_mot = upstream.rateeq(beams_obe, mag_quad15, H_obj)
    eq_mot.set_initial_position_and_velocity(RATEEQ_MOTION_R0, RATEEQ_MOTION_V0)
    eq_mot.set_initial_pop_from_equilibrium()

    sol_mot = eq_mot.evolve_motion(RATEEQ_MOTION_TIME["t_span"], t_eval=RATEEQ_MOTION_TIME["t_eval"])
    t_mot = time.perf_counter() - t0

    np.savez_compressed(
        os.path.join(data_dir, "rateeq_motion_ref.npz"),
        t=sol_mot.t,
        r=sol_mot.r,
        v=sol_mot.v,
        N=sol_mot.N,
        r0=RATEEQ_MOTION_R0,
        v0=RATEEQ_MOTION_V0,
        upstream_time=t_mot,
    )
    print(f"  -> Saved rateeq_motion_ref.npz (Upstream time: {t_mot:.4f}s)")
    # endregion

    # region OBE Density (Rabi Oscillations & Spontaneous Damping)
    print("\n[8/9] Running Upstream OBE evolve_density...")
    t0 = time.perf_counter()
    beam_rabi = upstream.laserBeam(**OBE_RABI_BEAM_CONFIG)
    beams_rabi = upstream.laserBeams([beam_rabi])

    obe_dens = upstream.obe(beams_rabi, mag_const0, H_obj, transform_into_re_im=True)
    obe_dens.set_initial_rho_equally()

    sol_dens = obe_dens.evolve_density(OBE_DENSITY_TIME["t_span"], t_eval=OBE_DENSITY_TIME["t_eval"])
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
    kvec,pol,s,delta = OBE_MOTION_BEAM_CONFIG.values()
    beam_obe_mot = upstream.laserBeam(kvec,pol.value,s,delta)
    beams_obe_mot = upstream.laserBeams([beam_obe_mot])

    obe_mot = upstream.obe(beams_obe_mot, mag_const0, H_obj, transform_into_re_im=True)
    obe_mot.set_initial_position_and_velocity(OBE_MOTION_R0, OBE_MOTION_V0)
    obe_mot.set_initial_rho_equally()

    sol_obe_mot = obe_mot.evolve_motion(OBE_MOTION_TIME["t_span"], t_eval=OBE_MOTION_TIME["t_eval"])
    t_obe_mot = time.perf_counter() - t0

    np.savez_compressed(
        os.path.join(data_dir, "obe_motion_ref.npz"),
        t=sol_obe_mot.t,
        r=sol_obe_mot.r,
        v=sol_obe_mot.v,
        rho=sol_obe_mot.rho,
        r0=OBE_MOTION_R0,
        v0=OBE_MOTION_V0,
        upstream_time=t_obe_mot,
    )
    print(f"  -> Saved obe_motion_ref.npz (Upstream time: {t_obe_mot:.4f}s)")
    # endregion

    print("\n" + "=" * 60)
    print(f"All upstream benchmark reference files generated successfully in "
          f"{time.perf_counter()-start_time:.4f} s!")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_benchmarks()
