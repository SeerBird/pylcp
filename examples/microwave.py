"""
Microwave and Raman Atom-Light Interaction in Rubidium-87 (D2 Line)
==================================================================

This module models a closed-loop Delta-system in Rubidium-87 using PyLCP's
DataFrame framework, utilizing the physical atomic parameters from Steck
(Rb87 D2 line, hyperfine structures, branching ratios, and transition dipoles).

System Configuration:
  - Ground clock state 1: |5^2S_{1/2}, F=1, m_F=0>
  - Ground clock state 2: |5^2S_{1/2}, F=2, m_F=0>
  - Excited state:       |5^2P_{3/2}, F'=2, m_F'=+1> (and optionally F'=1)
  - Couplings:
      * "Red" Laser  : drives |F=1, m_F=0> -> |F'=2, m_F'=+1> (sigma+)
      * "Blue" Laser : drives |F=2, m_F=0> -> |F'=2, m_F'=+1> (sigma+)
      * Microwave    : drives |F=1, m_F=0> <-> |F=2, m_F=0> (pi)

Simulation Cases:
  1) Case 1: Microwave Only (Lasers = 0, MW > 0)
     - Coherent clock Rabi oscillations and clock transition coherence.
  2) Case 2: Lasers Only (MW = 0, Lasers > 0)
     - Damped optical Rabi flopping, optical coherences, and dark-state
       (CPT) formation matching PyLCP Tutorial 07.
  3) Case 3: Combined System (Lasers > 0, MW > 0)
     - Closed-loop Delta-system dynamics over 100 us.
"""

from __future__ import annotations

import os
import time
import numpy as np

import matplotlib

from pylcp import OBE

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pylcp.atom.dataframe import DataFrame, PredefinedAtom
from pylcp.typing import PolarizationEnum, TransitionKey, HyperfineKey, HFTransitionKey

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_dataframe_system(
        s_red: float = 0.0,
        s_blue: float = 0.0,
        s_mw: float = 0.0,
        Delta: float = -2.0,
        delta: float = 0.0,
        mw_detuning: float = 0.0,
        include_F1_excited: bool = False) -> tuple[DataFrame, OBE]:
    """
    Constructs the Rubidium-87 system using PyLCP DataFrame.

    Parameters
    ----------
    s_red : float
        Saturation parameter of the "Red" probe laser (|F=1> -> |F'=2>).
    s_blue : float
        Saturation parameter of the "Blue" coupling laser (|F=2> -> |F'=2>).
    s_mw : float
        Saturation parameter of the microwave field (|F=1> <-> |F=2>).
    Delta : float
        Common optical detuning in units of Gamma (relative to excited state).
    delta : float
        Two-photon Raman detuning in units of Gamma.
    mw_detuning : float
        Microwave detuning in units of Gamma from clock transition resonance.
    include_F1_excited : bool
        If True, also includes the |F'=1, m_F'=+1> excited state (4-level model).
        If False, isolates the 3-level Lambda system (|F'=2, m_F'=+1>).

    Returns
    -------
    frame : DataFrame
    obe : pylcp.obe.OBE
    """
    # region Prep rubidium D1 line (the 3 states + optional F' = 1 state)
    frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
    g = 'g'
    e2 = 'e2'
    frame.remove_state('e1')  # D1 line not involved

    # Remove uncoupled hyperfine manifolds in 5^2P_{3/2}
    frame.remove_state(e2, 0)
    frame.remove_state(e2, 3)  # F'=3 decoupled from F=1 by dipole selection rule
    if not include_F1_excited:
        frame.remove_state(e2, 1)

    # Restrict to active magnetic sublevels:
    # Clock states m_F = 0 in ground manifolds
    frame.fine_states[g].substates[1] = [0]
    frame.fine_states[g].substates[2] = [0]

    # sigma+ light drives delta m = +1, coupling to m_F' = +1 in excited state(s)
    frame.fine_states[e2].substates[2] = [1]
    if include_F1_excited:
        frame.fine_states[e2].substates[1] = [1]

    # Register magnetic transition between ground manifolds (activates Hyperfine Block Mode)
    frame.add_magnetic_transition(g, 1, 2)
    # endregion
    # region Microwave Coupling between F=1 and F=2 (pi polarized)
    if s_mw > 0.0:
        frame.add_magnetic_laser(
            g, 1, 2,
            delta=mw_detuning,
            kvec=np.zeros(3),
            pol=np.array([0.0, 0.0, 1.0]),
            intensity=s_mw,
            delta_in_Gamma=True,
        )
    # endregion
    # region Optical Lasers (D2 line, sigma+ polarized)
    # "Red" laser: pre-determined to couple exclusively |g, F=1> -> |e2, F'=2>
    if s_red > 0.0:
        frame.add_laser_on_single_transition(
            g, e2, F1=1, F2=2,
            delta=Delta + delta,
            kvec=[0, 0, 1],
            pol=PolarizationEnum.SIGMA_PLUS,
            intensity=s_red,
            delta_in_Gamma=True,
        )

    # "Blue" laser: pre-determined to couple exclusively |g, F=2> -> |e2, F'=2>
    if s_blue > 0.0:
        frame.add_laser_on_single_transition(
            g, e2, F1=2, F2=2,
            delta=Delta,
            kvec=[0, 0, 1],
            pol=PolarizationEnum.SIGMA_PLUS,
            intensity=s_blue,
            delta_in_Gamma=True,
        )
    # endregion
    obe = frame.obe()
    return frame, obe


def run_case1_microwave_only(
        s_mw: float = 1.0,
        t_max_us: float = 4.0,
        num_points: int = 501,
        save_fig: bool = True,
):
    """
    Case 1: Lasers = 0, Microwave > 0.
    Simulates coherent clock Rabi oscillations between |F=1, m=0> and |F=2, m=0> using DataFrame.
    """
    print("\n" + "=" * 60, flush=True)
    print("Running Case 1: Microwave Only (Lasers = 0, MW > 0) [DataFrame]", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    frame, obe = build_dataframe_system(s_red=0.0, s_blue=0.0, s_mw=s_mw)

    ref_gamma, _ = frame._principal_gamma_and_energy()
    t_max_dimless = (t_max_us * 1e-6) * ref_gamma

    # Start 100% in |g_F1, m=0> (index 0)
    rho0 = np.zeros(obe.hamiltonian.n)
    rho0[0] = 1.0
    obe.set_initial_rho_from_populations(rho0)

    # Evaluation grid
    t_eval = np.linspace(0.0, t_max_dimless, num_points)
    sol = obe.evolve_density([0.0, t_max_dimless], t_eval=t_eval,rtol = 1e-9)
    dt = time.time() - t0

    t_us = (sol.t / ref_gamma) * 1e6
    pop_g1 = np.real(sol.rho[0, 0, :])
    pop_g2 = np.real(sol.rho[1, 1, :])
    coh_12 = sol.rho[0, 1, :]
    total_trace = np.sum([np.real(sol.rho[i, i, :]) for i in range(obe.hamiltonian.n)], axis=0)

    print(f"Case 1 completed in {dt:.3f} s ({len(sol.t)} points).", flush=True)
    print(f"Max population transfer to F=2, m=0: {np.max(pop_g2):.4f}", flush=True)
    print(f"Trace deviation: {np.max(np.abs(total_trace - 1.0)):.2e}", flush=True)

    if save_fig:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))
        # region Left: Populations
        ax1.plot(t_us, pop_g1, 'b-', linewidth=1.5, label=r'$|F=1, m_F=0\rangle$')
        ax1.plot(t_us, pop_g2, 'r--', linewidth=1.5, label=r'$|F=2, m_F=0\rangle$')
        ax1.plot(t_us, pop_g1 + pop_g2, 'k:', linewidth=1.0, label='Clock Sum')
        ax1.set_xlabel(r'Time ($\mu$s)')
        ax1.set_ylabel('Population')
        ax1.set_title(r'Clock Rabi Flopping ($s_{\mathrm{mw}} = ' + f'{s_mw:.1f}' + r'$)')
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc='center right')
        # endregion
        # region Right: Coherence
        ax2.plot(t_us, np.real(coh_12), 'b-', linewidth=1.2, label=r'Re$[\rho_{12}]$')
        ax2.plot(t_us, np.imag(coh_12), 'r--', linewidth=1.2, label=r'Im$[\rho_{12}]$')
        ax2.plot(t_us, np.abs(coh_12), 'k-', linewidth=1.5, label=r'$|\rho_{12}|$')
        ax2.set_xlabel(r'Time ($\mu$s)')
        ax2.set_ylabel(r'Coherence $\rho_{12}$')
        ax2.set_title('Clock Transition Coherence')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc='center right')
        # endregion
        fig.tight_layout()
        out_path = os.path.join(SCRIPT_DIR, 'case1_microwave_only.png')
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved figure to {out_path}", flush=True)

    return sol


def run_case2_lasers_only(
        s_red: float = 50.0,
        s_blue: float = 50.0,
        t_max_us: float = 100,
        Delta: float = -130.0,
        num_points: int = 1001,
        save_fig: bool = True,
):
    """
    Case 2: Microwave = 0, Lasers > 0.
    Simulates damped optical Rabi flopping, coherences, and dark-state
    formation (CPT), replicating PyLCP Tutorial 07 (cell [4]) using DataFrame.
    """
    print("\n" + "=" * 60, flush=True)
    print("Running Case 2: Lasers Only (MW = 0, Lasers > 0) [DataFrame]", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    frame, obe = build_dataframe_system(s_red=s_red, s_blue=s_blue, s_mw=0.0, Delta=Delta, delta=0.0)

    ref_gamma, _ = frame._principal_gamma_and_energy()
    t_max_dimless = (t_max_us * 1e-6) * ref_gamma
    # TODO: get convenience methods for units on frame
    #  or elsewhere

    # Start 100% in |g_F2, m=0> (index 1)
    rho0 = np.zeros(obe.hamiltonian.n)
    rho0[1] = 1.0
    obe.set_initial_rho_from_populations(rho0)

    t_eval = np.linspace(0.0, t_max_dimless, num_points)
    sol = obe.evolve_density([0.0, t_max_dimless], t_eval=t_eval,rtol = 1e-9)
    dt = time.time() - t0

    t_us = (sol.t / ref_gamma) * 1e6
    rho_g1 = np.real(sol.rho[0, 0, :])
    rho_g2 = np.real(sol.rho[1, 1, :])
    rho_e = np.real(sol.rho[2, 2, :])
    rho_12 = sol.rho[0, 1, :]
    rho_1e = sol.rho[0, 2, :]
    rho_2e = sol.rho[1, 2, :]

    print(f"Case 2 completed in {dt:.3f} s.", flush=True)
    print(f"Final populations: rho_11 = {rho_g1[-1]:.4f}, rho_22 = {rho_g2[-1]:.4f}", flush=True)
    print(f"Final excited population: rho_ee = {rho_e[-1]:.4e} (dark state trapping)", flush=True)
    print(f"Final Raman coherence: |rho_12| = {np.abs(rho_12[-1]):.4f}", flush=True)

    if save_fig:
        fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.75))

        # Left: Populations (replicating Tutorial 07)
        ax[0].plot(t_us, rho_g1, linewidth=0.75, label=r'$\rho_{gg}$')
        ax[0].plot(t_us, rho_g2, linewidth=0.75, label=r'$\rho_{rr}$')
        ax[0].plot(t_us, rho_e, linewidth=0.75, label=r'$\rho_{ee}$')
        ax[0].plot(t_us, rho_g1 + rho_g2 + rho_e, 'k--', linewidth=0.5)
        ax[0].legend(fontsize=7)
        ax[0].set_xlabel(r'Time ($\mu$s)')
        ax[0].set_ylabel(r'$\rho_{ii}$')
        ax[0].set_title(r'Damped Rabi Flopping & CPT', fontsize=9)

        # Right: Coherences (replicating Tutorial 07)
        ax[1].plot(t_us, np.real(rho_12), linewidth=0.75, label=r'Re$[\rho_{gr}]$')
        ax[1].plot(t_us, np.imag(rho_12), linewidth=0.75, label=r'Im$[\rho_{gr}]$')
        ax[1].plot(t_us, np.abs(rho_12), 'k-', linewidth=1.0, label=r'$|\rho_{gr}|$')
        ax[1].plot(t_us, np.abs(rho_1e), linewidth=0.75, label=r'$|\rho_{ge}|$')
        ax[1].plot(t_us, np.abs(rho_2e), linewidth=0.75, label=r'$|\rho_{re}|$')
        ax[1].legend(fontsize=7)
        ax[1].set_xlabel(r'Time ($\mu$s)')
        ax[1].set_ylabel(r'$\rho_{ij}$')
        ax[1].set_title(r'Dark-State Coherence Dynamics', fontsize=9)

        fig.subplots_adjust(wspace=0.15)
        fig.tight_layout()
        out_path = os.path.join(SCRIPT_DIR, 'case2_lasers_only_eit.png')
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved figure to {out_path}", flush=True)

    return sol


def run_case3_combined(
        s_red: float = 4.0,
        s_blue: float = 4.0,
        s_mw: float = 1.0,
        t_max_us: float = 100.0,
        Delta: float = -2.0,
        num_points: int = 5001,
        save_fig: bool = True,
):
    """
    Case 3: Combined System (Both Lasers > 0 AND Microwave > 0).
    Simulates the closed-loop Delta-system over 100 us using DataFrame.
    """
    print("\n" + "=" * 60, flush=True)
    print(f"Running Case 3: Closed-Loop Delta-System over {t_max_us:.1f} us [DataFrame]",
          flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    frame, obe = build_dataframe_system(
        s_red=s_red,
        s_blue=s_blue,
        s_mw=s_mw,
        Delta=Delta,
        delta=0.0,
    )

    ref_gamma, _ = frame._principal_gamma_and_energy()
    t_max_dimless = (t_max_us * 1e-6) * ref_gamma

    # Start 100% in |g_F1, m=0>
    rho0 = np.zeros(obe.hamiltonian.n)
    rho0[0] = 1.0
    obe.set_initial_rho_from_populations(rho0)

    # Dense grid (5001 points) across 100 us to resolve both fast transient and long-term equilibrium
    t_eval = np.linspace(0.0, t_max_dimless, num_points)
    sol = obe.evolve_density([0.0, t_max_dimless], t_eval=t_eval, rtol=1e-9, atol=1e-8)
    dt = time.time() - t0

    t_us = (sol.t / ref_gamma) * 1e6
    pop_g1 = np.real(sol.rho[0, 0, :])
    pop_g2 = np.real(sol.rho[1, 1, :])
    pop_e = np.real(sol.rho[2, 2, :])
    coh_12 = sol.rho[0, 1, :]
    total_trace = pop_g1 + pop_g2 + pop_e

    print(f"Case 3 (100 us) completed in {dt:.3f} s ({len(sol.t)} smooth points).", flush=True)
    print(f"Trace deviation: {np.max(np.abs(total_trace - 1.0)):.2e}", flush=True)
    print(
        f"Final populations: rho_11 = {pop_g1[-1]:.4f}, rho_22 = {pop_g2[-1]:.4f}, rho_ee = {pop_e[-1]:.4e}",
        flush=True)

    if save_fig:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))

        # Left Column: Zoom on initial transient (0 - 10 us)
        idx_zoom = t_us <= 10.0
        axes[0, 0].plot(t_us[idx_zoom], pop_g1[idx_zoom], 'b-', linewidth=1.2,
                        label=r'$\rho_{11}$ ($|F=1, m_F=0\rangle$)')
        axes[0, 0].plot(t_us[idx_zoom], pop_g2[idx_zoom], 'r--', linewidth=1.2,
                        label=r'$\rho_{22}$ ($|F=2, m_F=0\rangle$)')
        axes[0, 0].plot(t_us[idx_zoom], pop_e[idx_zoom] * 20, 'g-.', linewidth=1.2,
                        label=r'$\rho_{ee} \times 20$')
        axes[0, 0].set_ylabel('Population')
        axes[0, 0].set_title(r'Transient Dynamics ($0 - 10\ \mu\mathrm{s}$)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=7, loc='upper right')

        # Right Column: Full 100 us dynamics
        axes[0, 1].plot(t_us, pop_g1, 'b-', linewidth=1.2, label=r'$\rho_{11}$')
        axes[0, 1].plot(t_us, pop_g2, 'r--', linewidth=1.2, label=r'$\rho_{22}$')
        axes[0, 1].plot(t_us, pop_e * 20, 'g-.', linewidth=1.2, label=r'$\rho_{ee} \times 20$')
        axes[0, 1].set_title(r'Full Dynamics ($0 - 100\ \mu\mathrm{s}$)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=7, loc='center right')

        # Bottom-Left: Coherence Zoom (0 - 10 us)
        axes[1, 0].plot(t_us[idx_zoom], np.abs(coh_12[idx_zoom]), 'k-', linewidth=1.3,
                        label=r'$|\rho_{12}|$')
        axes[1, 0].plot(t_us[idx_zoom], np.real(coh_12[idx_zoom]), 'b:', linewidth=1.0,
                        label=r'Re$[\rho_{12}]$')
        axes[1, 0].plot(t_us[idx_zoom], np.imag(coh_12[idx_zoom]), 'r:', linewidth=1.0,
                        label=r'Im$[\rho_{12}]$')
        axes[1, 0].set_xlabel(r'Time ($\mu$s)')
        axes[1, 0].set_ylabel(r'Coherence $\rho_{12}$')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=7, loc='lower right')

        # Bottom-Right: Full 100 us Coherence
        axes[1, 1].plot(t_us, np.abs(coh_12), 'k-', linewidth=1.3, label=r'$|\rho_{12}|$')
        axes[1, 1].plot(t_us, np.real(coh_12), 'b:', linewidth=1.0, label=r'Re$[\rho_{12}]$')
        axes[1, 1].plot(t_us, np.imag(coh_12), 'r:', linewidth=1.0, label=r'Im$[\rho_{12}]$')
        axes[1, 1].set_xlabel(r'Time ($\mu$s)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=7, loc='center right')

        fig.tight_layout()
        out_path = os.path.join(SCRIPT_DIR, 'case3_combined_100us.png')
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved figure to {out_path}", flush=True)

    return sol


if __name__ == '__main__':
    print("=" * 60, flush=True)
    print("Rubidium-87 Microwave and Raman Simulation (DataFrame)", flush=True)
    print("=" * 60, flush=True)

    t_all_start = time.time()
    sol1 = run_case1_microwave_only()
    sol2 = run_case2_lasers_only()
    sol3 = run_case3_combined()
    total_time = time.time() - t_all_start

    print("\n" + "=" * 60, flush=True)
    print(f"All 3 DataFrame cases executed in {total_time:.2f} s!", flush=True)
    print("Generated figures:")
    print("  1) case1_microwave_only.png")
    print("  2) case2_lasers_only_eit.png (replicates Tutorial 07)")
    print("  3) case3_combined_100us.png (100 us Delta-system)")
    print("=" * 60, flush=True)
