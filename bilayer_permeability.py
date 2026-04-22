"""
Lipid Bilayer Permeability — Fick's Law Simulator
===================================================
Simulates passive diffusion of a solute (e.g., fluorescent dye,
small molecule drug) across a lipid bilayer membrane using Fick's
first law of diffusion.

Key Equations:
    Flux:    J = -P * (C_in - C_out)
    dC_in/dt = -(A / V) * P * (C_in - C_out)

    where:
      P = permeability coefficient (m/s)
      A = membrane surface area (m²)
      V = vesicle volume (m³)

This is directly relevant to:
    - Drug encapsulation efficiency in liposomes
    - Fluorescence leakage assays (calcein, ANTS/DPX)
    - Ion/solute transport across GUV membranes

Author: [Tamanna Mostafa
date:2026-04-20

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ── Vesicle geometry ─────────────────────────────────────────────────
R  = 10e-6          # m     GUV radius (10 µm)
A  = 4 * np.pi * R**2            # m²    surface area of sphere
V  = (4/3) * np.pi * R**3        # m³    volume of sphere

print(f"Vesicle radius    : {R*1e6:.1f} µm")
print(f"Surface area      : {A*1e12:.2f} µm²")
print(f"Volume            : {V*1e18:.2f} fL\n")

# ── Permeability coefficients (literature values) ────────────────────
# Different lipid compositions or pore-forming peptides change P
permeability_conditions = {
    "Pure DPPC bilayer"          : 1e-9,    # m/s  (gel phase, tight)
    "DOPC bilayer (fluid)"       : 5e-9,    # m/s  (fluid phase)
    "DOPC + 10% gramicidin"      : 1e-7,    # m/s  (ion channel pores)
    "DOPC + 30% cholesterol"     : 3e-9,    # m/s  (ordered phase)
}

# ── Initial concentrations ───────────────────────────────────────────
C_in_0  = 100e-3    # mol/m³   initial inside  (100 µM = 100e-6 mol/L = 100e-3 mol/m³)
C_out_0 = 0.0       # mol/m³   outside starts empty (leakage assay)

# ── Time span ────────────────────────────────────────────────────────
t_end  = 3600       # seconds (1 hour)
t_eval = np.linspace(0, t_end, 500)

# ── ODE: concentration change inside vesicle ─────────────────────────
def dC_dt(t, C, P, A, V, C_out):
    """
    ODE for solute concentration inside GUV over time.

    Parameters
    ----------
    t     : float   time (s)
    C     : list    [C_in] current inside concentration (mol/m³)
    P     : float   permeability coefficient (m/s)
    A     : float   membrane area (m²)
    V     : float   vesicle volume (m³)
    C_out : float   outside concentration (mol/m³, treated as constant reservoir)

    Returns
    -------
    dC_in/dt : list
    """
    flux = -P * (C[0] - C_out)           # Fick's first law
    return [(A / V) * flux]

# ── Solve and plot ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Lipid Bilayer Permeability — Solute Leakage from GUV", fontsize=14, fontweight="bold")

colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]
t_min  = t_eval / 60          # convert seconds → minutes for x-axis

for idx, (label, P) in enumerate(permeability_conditions.items()):
    sol = solve_ivp(
        fun    = dC_dt,
        t_span = (0, t_end),
        y0     = [C_in_0],
        t_eval = t_eval,
        args   = (P, A, V, C_out_0),
        method = "RK45"
    )

    C_in     = sol.y[0]
    retained = (C_in / C_in_0) * 100      # % of original still inside

    half_life = np.log(2) / (P * A / V)   # analytical t½ for first-order decay
    print(f"{label}")
    print(f"  P = {P:.1e} m/s  |  half-life = {half_life/60:.1f} min")

    axes[0].plot(t_min, C_in * 1e3, color=colors[idx], linewidth=2, label=label)
    axes[1].plot(t_min, retained,   color=colors[idx], linewidth=2, label=label)

print()

# ── Format left panel: concentration ─────────────────────────────────
axes[0].set_xlabel("Time (min)", fontsize=12)
axes[0].set_ylabel("Inside Concentration (µM)", fontsize=12)
axes[0].set_title("Solute Concentration Inside GUV", fontsize=12)
axes[0].legend(fontsize=8, loc="upper right")
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=0)

# ── Format right panel: % retention ─────────────────────────────────
axes[1].set_xlabel("Time (min)", fontsize=12)
axes[1].set_ylabel("Solute Retained (%)", fontsize=12)
axes[1].set_title("Leakage Assay — % Retained vs Time", fontsize=12)
axes[1].legend(fontsize=8, loc="upper right")
axes[1].grid(True, alpha=0.3)
axes[1].axhline(50, color="gray", linestyle="--", alpha=0.5, label="50% mark")
axes[1].set_ylim(0, 105)

plt.tight_layout()
plt.savefig("bilayer_permeability.png", dpi=150, bbox_inches="tight")
plt.show()

print("Done! Figure saved as bilayer_permeability.png")