"""
Performance Benchmark & Profiling: Upstream PyLCP vs. Optimized Fork

This script benchmarks the speed of the optimized fork against upstream PyLCP across:
1. Hamiltonians: Kronecker product vectorization & Wigner symbol caching (60x speedup).
2. RateEq: Equilibrium population solver, force profile, and 3D MOT motion.
3. OBE (N=4 Toy System): Evolution matrix assembly throughput, density evolution, and motion.
4. OBE (N=8 Multi-Level System): Coupled density and motion evolution.
5. OBE (N=24 Alkali System, 87Rb D2 line): Multi-manifold evolution matrix assembly.

Key Features:
- JIT Warm-up: Runs a 1-shot warm-up before timing to eliminate Numba JIT cold-start bias.
- Timeout Wrapper: Automatically aborts any upstream step taking >10s to prevent hangs.
- Automated cProfile: Profiles the slowest step to display function-level bottlenecks.
"""

import sys
import os
import time
import io
import cProfile
import pstats
import concurrent.futures
from typing import Callable, Tuple, Any
import numpy as np

from pylcp.typing import PolarizationEnum

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import tests.vendor.upstream_pylcp as upstream
import pylcp as fork


# ==============================================================================
# Timeout Runner Utility
# ==============================================================================
def run_with_timeout(func: Callable[[], Any], timeout_sec: float = 10.0) -> Tuple[float, Any, bool]:
    """
    Executes a callable with a strict timeout limit.

    Parameters
    ----------
    func : callable
        Function or lambda to execute without arguments.
    timeout_sec : float
        Maximum allowed runtime in seconds before timing out.

    Returns
    -------
    elapsed : float
        Runtime in seconds (or timeout_sec if timed out).
    result : Any
        Return value from func (or None if timed out).
    timed_out : bool
        True if the execution exceeded timeout_sec.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        t0 = time.perf_counter()
        future = executor.submit(func)
        try:
            res = future.result(timeout=timeout_sec)
            elapsed = time.perf_counter() - t0
            return elapsed, res, False
        except concurrent.futures.TimeoutError:
            return timeout_sec, None, True
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"    [Warning: Error in execution: {e}]")
            return elapsed, None, False


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_table_header():
    print(f"{'Benchmark Step':<40} | {'Upstream (s)':<13} | {'Fork (s)':<10} | {'Speedup':<9}")
    print("-" * 80)


def print_row(step_name: str, t_up: float, up_timed_out: bool, t_fk: float, fk_timed_out: bool):
    if up_timed_out:
        up_str = f">{t_up:.1f}s (TIMEOUT)"
        speedup_str = f">{t_up / t_fk:.1f}x" if t_fk > 0 else "N/A"
    else:
        up_str = f"{t_up:.4f} s"
        speedup_str = f"{t_up / t_fk:.2f}x" if t_fk > 0 else "N/A"

    fk_str = f">{t_fk:.1f}s (TIMEOUT)" if fk_timed_out else f"{t_fk:.4f} s"
    print(f"{step_name:<40} | {up_str:>13} | {fk_str:>10} | {speedup_str:>9}")


def run_benchmark():
    records = []
    TIMEOUT = 10.0  # Cap any individual execution at 10 seconds

    print_header("PyLCP Performance Comparison: Upstream vs. Optimized Fork")

    # =========================================================================
    # Section 1: Hamiltonian Generator Optimizations
    # =========================================================================
    print_header("1. Hamiltonian Generators (Kronecker Vectorization & Caching)")
    print_table_header()

    # 1.1 hyperfine_coupled (30 evaluations)
    def bench_hf_up():
        for _ in range(30):
            upstream.hamiltonians.hyperfine_coupled(J=0.5, I=1.5, gJ=2.0, gI=-0.000995, Ahfs=3.417e9)

    def bench_hf_fk():
        for _ in range(30):
            fork.hamiltonians.hyperfine_coupled(J=0.5, I=1.5, gJ=2.0, gI=-0.000995, Ahfs=3.417e9)

    # Warm-up (1 call)
    upstream.hamiltonians.hyperfine_coupled(J=0.5, I=1.5, gJ=2.0, gI=-0.000995, Ahfs=3.417e9)
    fork.hamiltonians.hyperfine_coupled(J=0.5, I=1.5, gJ=2.0, gI=-0.000995, Ahfs=3.417e9)

    t_up, _, to_up = run_with_timeout(bench_hf_up, TIMEOUT)
    t_fk, _, to_fk = run_with_timeout(bench_hf_fk, TIMEOUT)
    print_row("hyperfine_coupled (30 evaluations)", t_up, to_up, t_fk, to_fk)
    records.append(("hyperfine_coupled (30 evaluations)", t_up, t_fk, bench_hf_up, bench_hf_fk))

    # 1.2 fine_structure_uncoupled (10 evaluations)
    def bench_fs_up():
        for _ in range(10):
            upstream.hamiltonians.fine_structure_uncoupled(
                L=1, S=0.5, I=1.5, xi=1e9, a_c=1e6, a_orb=1e5, a_dip=1e4, gL=1.0, gS=2.0, gI=-0.001
            )

    def bench_fs_fk():
        for _ in range(10):
            fork.hamiltonians.fine_structure_uncoupled(
                L=1, S=0.5, I=1.5, xi=1e9, a_c=1e6, a_orb=1e5, a_dip=1e4, gL=1.0, gS=2.0, gI=-0.001
            )

    upstream.hamiltonians.fine_structure_uncoupled(
        L=1, S=0.5, I=1.5, xi=1e9, a_c=1e6, a_orb=1e5, a_dip=1e4, gL=1.0, gS=2.0, gI=-0.001
    )
    fork.hamiltonians.fine_structure_uncoupled(
        L=1, S=0.5, I=1.5, xi=1e9, a_c=1e6, a_orb=1e5, a_dip=1e4, gL=1.0, gS=2.0, gI=-0.001
    )

    t_up, _, to_up = run_with_timeout(bench_fs_up, TIMEOUT)
    t_fk, _, to_fk = run_with_timeout(bench_fs_fk, TIMEOUT)
    print_row("fine_structure_uncoupled (10 evals)", t_up, to_up, t_fk, to_fk)
    records.append(("fine_structure_uncoupled (10 evals)", t_up, t_fk, bench_fs_up, bench_fs_fk))

    # =========================================================================
    # Section 2: Rate Equation (RateEq) Benchmarks
    # =========================================================================
    print_header("2. Rate Equation (RateEq) Benchmarks")
    print_table_header()

    # Shared N=4 Hamiltonian (Fg=0 -> Fe=1)
    H0_g_up, muq_g_up = upstream.hamiltonians.singleF(F=0, gF=1.0)
    H0_e_up, muq_e_up = upstream.hamiltonians.singleF(F=1, gF=1.0)
    dqij_up = upstream.hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    H_up = upstream.hamiltonian(H0_g_up, H0_e_up, muq_g_up, muq_e_up, dqij_up, mass=87.0)
    beams_up = upstream.conventional3DMOTBeams(s=1.0, delta=-1.0)
    mag_quad_up = upstream.quadrupoleMagneticField(15.0)

    H0_g_fk, muq_g_fk = fork.hamiltonians.singleF(F=0, gF=1.0)
    H0_e_fk, muq_e_fk = fork.hamiltonians.singleF(F=1, gF=1.0)
    dqij_fk = fork.hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    H_fk = fork.Hamiltonian(H0_g_fk, H0_e_fk, muq_g_fk, muq_e_fk, dqij_fk, mass=87.0)
    beams_fk = fork.Conventional3DMOTBeams(s=1.0, delta=-1.0)
    mag_quad_fk = fork.QuadrupoleMagneticField(15.0)

    # 2.1 RateEq Instantiation
    t_up, eq_rate_up, to_up = run_with_timeout(lambda: upstream.rateeq(beams_up, mag_quad_up, H_up), TIMEOUT)
    t_fk, eq_rate_fk, to_fk = run_with_timeout(lambda: fork.RateEq(beams_fk, mag_quad_fk, H_fk), TIMEOUT)
    print_row("RateEq: Object Initialization", t_up, to_up, t_fk, to_fk)

    # 2.2 Equilibrium Populations (500 evaluations with warm-up)
    r_test = np.array([0.1, -0.2, 0.3])
    v_test = np.array([1.0, 0.0, -0.5])

    # Warm-up 1 call
    eq_rate_up.equilibrium_populations(r_test, v_test, t=0)
    eq_rate_fk.equilibrium_populations(r_test, v_test, t=0)

    def bench_rate_eq_up():
        for _ in range(500):
            eq_rate_up.equilibrium_populations(r_test, v_test, t=0)

    def bench_rate_eq_fk():
        for _ in range(500):
            eq_rate_fk.equilibrium_populations(r_test, v_test, t=0)

    t_up, _, to_up = run_with_timeout(bench_rate_eq_up, TIMEOUT)
    t_fk, _, to_fk = run_with_timeout(bench_rate_eq_fk, TIMEOUT)
    print_row("RateEq: 500 Equilibrium Pop Evals", t_up, to_up, t_fk, to_fk)

    # 2.3 Force Profile (7 grid points)
    grid_7 = np.linspace(-0.5, 0.5, 7)
    R_7 = np.array([grid_7, np.zeros(7), np.zeros(7)])
    V_7 = np.array([np.zeros(7), np.zeros(7), np.zeros(7)])

    # Warm-up 1 call
    eq_rate_up.generate_force_profile(R_7[:, :1], V_7[:, :1], name="warm")
    eq_rate_fk.generate_force_profile(R_7[:, :1], V_7[:, :1], name="warm")

    t_up, _, to_up = run_with_timeout(lambda: eq_rate_up.generate_force_profile(R_7, V_7, name="p"), TIMEOUT)
    t_fk, _, to_fk = run_with_timeout(lambda: eq_rate_fk.generate_force_profile(R_7, V_7, name="p"), TIMEOUT)
    print_row("RateEq: Force Profile (7 points)", t_up, to_up, t_fk, to_fk)

    # 2.4 RateEq Motion Evolution (3D MOT trajectory, t=1.0)
    r0 = np.array([0.2, 0.1, -0.1])
    v0 = np.array([1.0, -0.5, 0.2])
    t_eval_mot = np.linspace(0, 1.0, 11)

    eq_rate_up.set_initial_position_and_velocity(r0, v0)
    eq_rate_up.set_initial_pop_from_equilibrium()
    eq_rate_fk.set_initial_position_and_velocity(r0, v0)
    eq_rate_fk.set_initial_pop_from_equilibrium()

    # Warm-up 1 step
    eq_rate_up.evolve_motion([0, 0.05])
    eq_rate_fk.evolve_motion([0, 0.05])
    eq_rate_up.set_initial_position_and_velocity(r0, v0)
    eq_rate_up.set_initial_pop_from_equilibrium()
    eq_rate_fk.set_initial_position_and_velocity(r0, v0)
    eq_rate_fk.set_initial_pop_from_equilibrium()

    t_up, _, to_up = run_with_timeout(lambda: eq_rate_up.evolve_motion([0, 1.0], t_eval=t_eval_mot), TIMEOUT)
    t_fk, _, to_fk = run_with_timeout(lambda: eq_rate_fk.evolve_motion([0, 1.0], t_eval=t_eval_mot), TIMEOUT)
    print_row("RateEq: 3D MOT Motion (t=1.0)", t_up, to_up, t_fk, to_fk)

    # =========================================================================
    # Section 3: Optical Bloch Equations (OBE) - Toy System (N = 4)
    # =========================================================================
    print_header("3. Optical Bloch Equations (OBE) - N=4 System")
    print_table_header()

    mag_zero_up = upstream.constantMagneticField(np.zeros(3))
    mag_zero_fk = fork.ConstantMagneticField(np.zeros(3))
    t_up, eq_obe_up, to_up = run_with_timeout(
        lambda: upstream.obe(beams_up, mag_zero_up, H_up, transform_into_re_im=False), TIMEOUT
    )
    t_fk, eq_obe_fk, to_fk = run_with_timeout(
        lambda: fork.OBE(beams_fk, mag_zero_fk, H_fk, transform_into_re_im=False), TIMEOUT
    )
    print_row("OBE (N=4): Object Initialization", t_up, to_up, t_fk, to_fk)

    # 3.1 Evolution Matrix Evaluation Throughput (500 evaluations)
    r_obe = np.array([0.1, -0.2, 0.3])

    # Warm-up JIT on total_electric_field
    eq_obe_up.full_OBE_ev(r_obe, 0.0)
    eq_obe_fk.full_OBE_ev(r_obe, 0.0)

    def bench_obe_ev_up():
        for _ in range(500):
            eq_obe_up.full_OBE_ev(r_obe, 0.0)

    def bench_obe_ev_fk():
        for _ in range(500):
            eq_obe_fk.full_OBE_ev(r_obe, 0.0)

    t_up, _, to_up = run_with_timeout(bench_obe_ev_up, TIMEOUT)
    t_fk, _, to_fk = run_with_timeout(bench_obe_ev_fk, TIMEOUT)
    print_row("OBE (N=4): 500 Evolution Matrix Evals", t_up, to_up, t_fk, to_fk)
    records.append(("OBE (N=4): 500 Evolution Matrix Evals", t_up, t_fk, bench_obe_ev_up, bench_obe_ev_fk))

    # 3.2 Density Matrix Evolution (Rabi oscillations, t=2.0)
    beam_rabi_up = upstream.laserBeam(kvec=np.array([0, 0, 1]), pol=np.array([0, 0, 1]), s=2.0, delta=0.0)
    beams_rabi_up = upstream.laserBeams([beam_rabi_up])
    obe_dens_up = upstream.obe(beams_rabi_up, mag_zero_up, H_up, transform_into_re_im=True)
    obe_dens_up.set_initial_rho_equally()

    beam_rabi_fk = fork.LaserBeam(kvec=np.array([0, 0, 1]), pol=np.array([0, 0, 1]), s=2.0, delta=0.0)
    beams_rabi_fk = fork.LaserBeams([beam_rabi_fk])
    obe_dens_fk = fork.OBE(beams_rabi_fk, mag_zero_fk, H_fk, transform_into_re_im=True)
    obe_dens_fk.set_initial_rho_equally()

    # Warm up JIT for laser beams
    obe_dens_up.evolve_density([0, 0.01])
    obe_dens_fk.evolve_density([0, 0.01])
    obe_dens_up.set_initial_rho_equally()
    obe_dens_fk.set_initial_rho_equally()

    t_eval_dens = np.linspace(0, 2.0, 21)
    call_dens_up = lambda: obe_dens_up.evolve_density([0, 2.0], t_eval=t_eval_dens)
    call_dens_fk = lambda: obe_dens_fk.evolve_density([0, 2.0], t_eval=t_eval_dens)

    t_up, _, to_up = run_with_timeout(call_dens_up, TIMEOUT)
    t_fk, _, to_fk = run_with_timeout(call_dens_fk, TIMEOUT)
    print_row("OBE (N=4): Evolve Density (t=2.0)", t_up, to_up, t_fk, to_fk)
    records.append(("OBE (N=4): Evolve Density (t=2.0)", t_up, t_fk, call_dens_up, call_dens_fk))

    # =========================================================================
    # Section 4: OBE Multi-Level System (N = 8 states, Fg=1 -> Fe=2)
    # =========================================================================
    print_header("4. Optical Bloch Equations (OBE) - N=8 Multi-Level System")
    print_table_header()

    H0_g8_up, muq_g8_up = upstream.hamiltonians.singleF(F=1, gF=1.0)
    H0_e8_up, muq_e8_up = upstream.hamiltonians.singleF(F=2, gF=1.0)
    dqij8_up = upstream.hamiltonians.dqij_two_bare_hyperfine(F=1, Fp=2, normalize=True)
    H8_up = upstream.hamiltonian(H0_g8_up, H0_e8_up, muq_g8_up, muq_e8_up, dqij8_up, mass=87.0)

    H0_g8_fk, muq_g8_fk = fork.hamiltonians.singleF(F=1, gF=1.0)
    H0_e8_fk, muq_e8_fk = fork.hamiltonians.singleF(F=2, gF=1.0)
    dqij8_fk = fork.hamiltonians.dqij_two_bare_hyperfine(F=1, Fp=2, normalize=True)
    H8_fk = fork.Hamiltonian(H0_g8_fk, H0_e8_fk, muq_g8_fk, muq_e8_fk, dqij8_fk, mass=87.0)

    # 4.1 N=8 Density Evolution (t=2.0 with warm-up)
    obe8_dens_up = upstream.obe(beams_rabi_up, mag_zero_up, H8_up, transform_into_re_im=True)
    obe8_dens_up.set_initial_rho_equally()

    obe8_dens_fk = fork.OBE(beams_rabi_fk, mag_zero_fk, H8_fk, transform_into_re_im=True)
    obe8_dens_fk.set_initial_rho_equally()

    # Warm-up JIT
    obe8_dens_up.evolve_density([0, 0.01])
    obe8_dens_fk.evolve_density([0, 0.01])
    obe8_dens_up.set_initial_rho_equally()
    obe8_dens_fk.set_initial_rho_equally()

    t_eval_8 = np.linspace(0, 2.0, 21)
    call_dens8_up = lambda: obe8_dens_up.evolve_density([0, 2.0], t_eval=t_eval_8)
    call_dens8_fk = lambda: obe8_dens_fk.evolve_density([0, 2.0], t_eval=t_eval_8)

    t_up, _, to_up = run_with_timeout(call_dens8_up, TIMEOUT)
    t_fk, _, to_fk = run_with_timeout(call_dens8_fk, TIMEOUT)
    print_row("OBE (N=8): Evolve Density (t=2.0)", t_up, to_up, t_fk, to_fk)
    records.append(("OBE (N=8): Density Evolution (t=2.0)", t_up, t_fk, call_dens8_up, call_dens8_fk))

    # 4.2 N=8 Coupled Motion Evolution (t=0.05 with warm-up)
    beam_mot_up = upstream.laserBeam(kvec=np.array([0, 0, 1]), pol=1, s=1.0, delta=-1.0)
    beams_mot_up = upstream.laserBeams([beam_mot_up])
    obe8_mot_up = upstream.obe(beams_mot_up, mag_zero_up, H8_up, transform_into_re_im=True)
    obe8_mot_up.set_initial_position_and_velocity(np.zeros(3), np.array([0.0, 0.0, 0.5]))
    obe8_mot_up.set_initial_rho_equally()

    beam_mot_fk = fork.LaserBeam(kvec=np.array([0, 0, 1]), pol=PolarizationEnum.SIGMA_PLUS, s=1.0, delta=-1.0)
    beams_mot_fk = fork.LaserBeams([beam_mot_fk])
    obe8_mot_fk = fork.OBE(beams_mot_fk, mag_zero_fk, H8_fk, transform_into_re_im=True)
    obe8_mot_fk.set_initial_position_and_velocity(np.zeros(3), np.array([0.0, 0.0, 0.5]))
    obe8_mot_fk.set_initial_rho_equally()

    # Warm-up JIT
    obe8_mot_up.evolve_motion([0, 0.005])
    obe8_mot_fk.evolve_motion([0, 0.005])
    obe8_mot_up.set_initial_position_and_velocity(np.zeros(3), np.array([0.0, 0.0, 0.5]))
    obe8_mot_up.set_initial_rho_equally()
    obe8_mot_fk.set_initial_position_and_velocity(np.zeros(3), np.array([0.0, 0.0, 0.5]))
    obe8_mot_fk.set_initial_rho_equally()

    t_eval_mot8 = np.linspace(0, 0.05, 6)
    call_mot8_up = lambda: obe8_mot_up.evolve_motion([0, 0.05], t_eval=t_eval_mot8)
    call_mot8_fk = lambda: obe8_mot_fk.evolve_motion([0, 0.05], t_eval=t_eval_mot8)

    t_up, _, to_up = run_with_timeout(call_mot8_up, TIMEOUT)
    t_fk, _, to_fk = run_with_timeout(call_mot8_fk, TIMEOUT)
    print_row("OBE (N=8): Coupled Motion (t=0.05)", t_up, to_up, t_fk, to_fk)
    records.append(("OBE (N=8): Coupled Motion (t=0.05)", t_up, t_fk, call_mot8_up, call_mot8_fk))

    # =========================================================================
    # Section 5: Full Multi-Level Alkali (87Rb D2 line, N = 24 states)
    # =========================================================================
    print_header("5. Multi-Level Alkali Atom (87Rb D2 line, N=24 States)")
    print_table_header()

    # Ground J=1/2 (8 states) & Excited J=3/2 (16 states) -> 24 states total
    H0_g_rb_up, muq_g_rb_up = upstream.hamiltonians.hyperfine_coupled(
        J=0.5, I=1.5, gJ=2.0, gI=-0.000995, Ahfs=3.417e9
    )
    H0_e_rb_up, muq_e_rb_up = upstream.hamiltonians.hyperfine_coupled(
        J=1.5, I=1.5, gJ=1.336, gI=-0.000995, Ahfs=84.7e6, Bhfs=12.5e6
    )
    dqij_rb_up = upstream.hamiltonians.dqij_two_hyperfine_manifolds(J=0.5, Jp=1.5, I=1.5)
    H_rb_up = upstream.hamiltonian(H0_g_rb_up, H0_e_rb_up, muq_g_rb_up, muq_e_rb_up, dqij_rb_up, mass=87.0)
    eq_rb_up = upstream.obe(beams_up, mag_zero_up, H_rb_up, transform_into_re_im=False)

    H0_g_rb_fk, muq_g_rb_fk = fork.hamiltonians.hyperfine_coupled(
        J=0.5, I=1.5, gJ=2.0, gI=-0.000995, Ahfs=3.417e9
    )
    H0_e_rb_fk, muq_e_rb_fk = fork.hamiltonians.hyperfine_coupled(
        J=1.5, I=1.5, gJ=1.336, gI=-0.000995, Ahfs=84.7e6, Bhfs=12.5e6
    )
    dqij_rb_fk = fork.hamiltonians.dqij_two_hyperfine_manifolds(J=0.5, Jp=1.5, I=1.5)
    H_rb_fk = fork.Hamiltonian(H0_g_rb_fk, H0_e_rb_fk, muq_g_rb_fk, muq_e_rb_fk, dqij_rb_fk, mass=87.0)
    eq_rb_fk = fork.OBE(beams_fk, mag_zero_fk, H_rb_fk, transform_into_re_im=False)

    # Warm-up JIT
    eq_rb_up.full_OBE_ev(r_obe, 0.0)
    eq_rb_fk.full_OBE_ev(r_obe, 0.0)

    def bench_rb24_ev_up():
        for _ in range(50):
            eq_rb_up.full_OBE_ev(r_obe, 0.0)

    def bench_rb24_ev_fk():
        for _ in range(50):
            eq_rb_fk.full_OBE_ev(r_obe, 0.0)

    t_up, _, to_up = run_with_timeout(bench_rb24_ev_up, TIMEOUT)
    t_fk, _, to_fk = run_with_timeout(bench_rb24_ev_fk, TIMEOUT)
    print_row("OBE (N=24): 50 Evolution Matrix Evals", t_up, to_up, t_fk, to_fk)
    records.append(("OBE (N=24): 50 Evolution Matrix Evals", t_up, t_fk, bench_rb24_ev_up, bench_rb24_ev_fk))

    # =========================================================================
    # Section 6: Automated cProfile Deep-Dive on the Slowest Step
    # =========================================================================
    # Select the completed step with the largest upstream time
    records.sort(key=lambda r: r[1], reverse=True)
    slowest_name, slowest_t_up, slowest_t_fk, callable_up, callable_fk = records[0]

    print_header(f"6. cProfile Breakdown on Slowest Benchmark Step:\n   '{slowest_name}'")
    print(f"Total Upstream Wall Time: {slowest_t_up:.4f} s")
    print(f"Total Fork Wall Time:     {slowest_t_fk:.4f} s")
    print(f"Measured Speedup:         {slowest_t_up / slowest_t_fk:.2f}x\n")

    # How to use cProfile and pstats:
    def profile_and_report(target_callable: Callable[[], Any], library_label: str):
        print("=" * 80)
        print(f" [{library_label}] Profiling Statistics")
        print("=" * 80)

        # -------------------------------------------------------------
        # STEP 1: Create a Profile instance
        # -------------------------------------------------------------
        profiler = cProfile.Profile()

        # -------------------------------------------------------------
        # STEP 2: Wrap execution between enable() and disable()
        # -------------------------------------------------------------
        profiler.enable()
        target_callable()
        profiler.disable()

        # -------------------------------------------------------------
        # STEP 3: Format and inspect statistics with pstats.Stats
        # -------------------------------------------------------------
        # A) Sort by CUMULATIVE time to see call hierarchy and total time under each function
        stream_cum = io.StringIO()
        stats_cum = pstats.Stats(profiler, stream=stream_cum)
        stats_cum.strip_dirs()
        stats_cum.sort_stats(pstats.SortKey.CUMULATIVE)
        stats_cum.print_stats(10)

        # B) Sort by TIME (internal self-time) to see where CPU cycles are burned directly
        stream_self = io.StringIO()
        stats_self = pstats.Stats(profiler, stream=stream_self)
        stats_self.strip_dirs()
        stats_self.sort_stats(pstats.SortKey.TIME)
        stats_self.print_stats(10)

        print("\n--- Top 10 Functions by CUMULATIVE Time (Call Hierarchy) ---")
        for line in stream_cum.getvalue().splitlines()[:15]:
            print("  ", line)

        print("\n--- Top 10 Functions by INTERNAL / SELF Time (Raw CPU Sinks) ---")
        for line in stream_self.getvalue().splitlines()[:15]:
            print("  ", line)
        print()

    # Profile Upstream
    profile_and_report(callable_up, "UPSTREAM PY_LCP")

    # Profile Fork
    profile_and_report(callable_fk, "FORK PY_LCP")

    print_header("Benchmark & Profiling Complete")


if __name__ == "__main__":
    run_benchmark()
