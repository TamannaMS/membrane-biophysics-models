"""
Electroporation Pore Formation & Resealing Kinetics
=====================================================
Models how electropores open during a pulse and reseal
after the field is removed.

Two-phase model:
  Phase 1 — PORE OPENING (during pulse, 0 → t_pulse):
      dN/dt = alpha * exp(V_tm / V0) - N / tau_open

  Phase 2 — PORE RESEALING (after pulse, t_pulse → end):
      dN/dt = -N / tau_reseal

  where:
    N          = number of pores
    alpha      = pore creation rate constant (pores/s)
    V_tm       = transmembrane voltage (V)
    V0         = characteristic voltage (V)
    tau_open   = pore lifetime during pulse (s)
    tau_reseal = resealing time constant (s)

Membrane conductance from pores:
    G_pore(t) = N(t) * g_single
    where g_single = conductance of one pore (~1 nS typical)

date:2026-04-21
Reference: Weaver & Chizmadzhev (1996). Bioelectrochemistry and Bioenergetics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ── Pore kinetics parameters ─────────────────────────────────────────
alpha      = 1e9      # pores/s   creation rate prefactor
V0         = 0.25     # V         characteristic voltage
tau_open   = 1e-6     # s         pore lifetime during pulse (1 µs)
tau_reseal = 1e-3     # s         resealing time constant (1 ms)

# ── Single pore conductance ──────────────────────────────────────────
g_single   = 1e-9     # S         ~1 nS per pore (literature)

# ── Pulse parameters ────────────────────────────────────────────────
t_pulse    = 100e-6   # s         pulse duration (100 µs)
t_total    = 5e-3     # s         total simulation time (5 ms)

# ── Transmembrane voltages to compare ───────────────────────────────
V_tm_values = [0.3, 0.5, 0.8, 1.0]    # V

# ── Time arrays ──────────────────────────────────────────────────────
t_during = np.linspace(0, t_pulse, 200)                    # during pulse
t_after  = np.linspace(t_pulse, t_total, 300)              # after pulse
t_full   = np.concatenate([t_during, t_after[1:]])         # full timeline

# ── ODE: pore number during pulse ────────────────────────────────────
def dN_dt_open(t, N, alpha, V_tm, V0, tau_open):
    """Pore creation ODE during electric pulse."""
    creation = alpha * np.exp(V_tm / V0)
    decay    = N[0] / tau_open
    return [creation - decay]

# ── ODE: pore number during resealing ───────────────────────────────
def dN_dt_reseal(t, N, tau_reseal):
    """Pore closure ODE after pulse ends."""
    return [-N[0] / tau_reseal]

# ── Plot setup ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Electroporation — Pore Formation & Resealing Kinetics",
             fontsize=14, fontweight="bold")

colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(V_tm_values)))

all_N_peak   = []
all_G_peak   = []

for i, V_tm in enumerate(V_tm_values):

    # ── Phase 1: pore opening during pulse ──────────────────────────
    sol_open = solve_ivp(
        fun    = dN_dt_open,
        t_span = (0, t_pulse),
        y0     = [0.0],
        t_eval = t_during,
        args   = (alpha, V_tm, V0, tau_open),
        method = "RK45"
    )
    N_open = sol_open.y[0]
    N_peak = N_open[-1]           # pore count at end of pulse

    # ── Phase 2: resealing after pulse ──────────────────────────────
    sol_reseal = solve_ivp(
        fun    = dN_dt_reseal,
        t_span = (t_pulse, t_total),
        y0     = [N_peak],
        t_eval = t_after,
        args   = (tau_reseal,),
        method = "RK45"
    )
    N_reseal = sol_reseal.y[0]

    # ── Combine phases ───────────────────────────────────────────────
    N_total = np.concatenate([N_open, N_reseal[1:]])
    G_total = N_total * g_single * 1e9    # convert S → nS

    t_ms = t_full * 1e3                   # convert s → ms

    # Store peak values
    all_N_peak.append(N_peak)
    all_G_peak.append(N_peak * g_single * 1e9)

    lbl = f"V_tm = {V_tm} V"

    # Top-left: pore number vs time
    axes[0, 0].plot(t_ms, N_total, color=colors[i], linewidth=2, label=lbl)

    # Top-right: membrane conductance vs time
    axes[0, 1].plot(t_ms, G_total, color=colors[i], linewidth=2, label=lbl)

    print(f"V_tm = {V_tm} V  →  Peak pores = {N_peak:.0f}  |  "
          f"Peak G = {N_peak*g_single*1e9:.2f} nS")

# ── Bottom-left: peak pore count vs V_tm ────────────────────────────
axes[1, 0].bar(range(len(V_tm_values)), all_N_peak,
               color=colors, alpha=0.85)
axes[1, 0].set_xticks(range(len(V_tm_values)))
axes[1, 0].set_xticklabels([f"{v} V" for v in V_tm_values])
axes[1, 0].set_xlabel("Transmembrane Voltage (V)", fontsize=12)
axes[1, 0].set_ylabel("Peak Pore Count", fontsize=12)
axes[1, 0].set_title("Peak Pores vs TMV", fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis="y")

# ── Bottom-right: peak conductance vs V_tm ──────────────────────────
axes[1, 1].bar(range(len(V_tm_values)), all_G_peak,
               color=colors, alpha=0.85)
axes[1, 1].set_xticks(range(len(V_tm_values)))
axes[1, 1].set_xticklabels([f"{v} V" for v in V_tm_values])
axes[1, 1].set_xlabel("Transmembrane Voltage (V)", fontsize=12)
axes[1, 1].set_ylabel("Peak Membrane Conductance (nS)", fontsize=12)
axes[1, 1].set_title("Peak Conductance vs TMV", fontsize=12)
axes[1, 1].grid(True, alpha=0.3, axis="y")

# ── Add pulse marker to time-series plots ───────────────────────────
for ax in [axes[0, 0], axes[0, 1]]:
    ax.axvline(t_pulse * 1e3, color="black", linestyle="--",
               linewidth=1.2, alpha=0.6, label="Pulse end")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (ms)", fontsize=12)

axes[0, 0].set_ylabel("Number of Pores", fontsize=12)
axes[0, 0].set_title("Pore Count over Time", fontsize=12)

axes[0, 1].set_ylabel("Membrane Conductance (nS)", fontsize=12)
axes[0, 1].set_title("Membrane Conductance over Time", fontsize=12)

plt.tight_layout()
plt.savefig("electroporation_pore_kinetics.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nParameters used:")
print(f"  Pulse duration  : {t_pulse*1e6:.0f} µs")
print(f"  Resealing τ     : {tau_reseal*1e3:.1f} ms")
print(f"  Single pore g   : {g_single*1e9:.0f} nS")
print("Figure saved as electroporation_pore_kinetics.png")