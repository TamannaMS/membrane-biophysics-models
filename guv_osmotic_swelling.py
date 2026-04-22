"""
GUV Osmotic Swelling Model
===========================
Models how Giant Unilamellar Vesicles (GUVs) swell over time
when placed in a hypotonic solution, based on osmotic pressure
and membrane permeability (Lp).

Key Equation:
    dR/dt = Lp * R0 * delta_pi / 3
    where delta_pi = osmotic pressure difference (Van't Hoff: delta_pi = RT * delta_C)

Reference concepts:
    - Van't Hoff osmotic pressure
    - Membrane water permeability (Lp)
    - Volume conservation in spherical vesicles

Author: [Tamanna Mostafa]
date:2026-04-20

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ── Physical constants ──────────────────────────────────────────────
R_gas = 8.314       # J / (mol·K)   universal gas constant
T     = 310.15      # K             physiological temperature (37°C)

# ── Membrane parameters ─────────────────────────────────────────────
Lp = 2e-13          # m / (Pa·s)    water permeability of lipid bilayer (DOPC typical)
R0 = 10e-6          # m             initial GUV radius (10 µm)

# ── Osmotic conditions ──────────────────────────────────────────────
# delta_C = concentration difference (outside - inside), mol/m³
# Positive = hypotonic outside → vesicle swells
delta_C_values = [0.5, 1.0, 2.0, 5.0]   # mol/m³  (= mM roughly)

# ── Time span ───────────────────────────────────────────────────────
t_start = 0
t_end   = 60        # seconds
t_eval  = np.linspace(t_start, t_end, 300)

# ── ODE: rate of radius change ───────────────────────────────────────
def dR_dt(t, R, Lp, delta_pi):
    """
    ODE for GUV radius over time under osmotic pressure.

    Parameters
    ----------
    t        : float  current time (s)
    R        : list   current radius [R] (m)
    Lp       : float  membrane water permeability (m/Pa/s)
    delta_pi : float  osmotic pressure difference (Pa)

    Returns
    -------
    dR/dt : list
    """
    return [Lp * delta_pi * R0 / (3 * R[0])]

# ── Solve and plot ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("GUV Osmotic Swelling Model", fontsize=14, fontweight="bold")

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(delta_C_values)))

for i, dC in enumerate(delta_C_values):
    # Van't Hoff equation: π = R_gas * T * C
    delta_pi = R_gas * T * dC          # osmotic pressure in Pa

    # Solve ODE
    sol = solve_ivp(
        fun      = dR_dt,
        t_span   = (t_start, t_end),
        y0       = [R0],               # initial radius
        t_eval   = t_eval,
        args     = (Lp, delta_pi),
        method   = "RK45"
    )

    R_um = sol.y[0] * 1e6             # convert m → µm
    t_s  = sol.t

    # Left plot: radius vs time
    axes[0].plot(t_s, R_um, color=colors[i], linewidth=2,
                 label=f"ΔC = {dC} mol/m³")

    # Right plot: relative swelling ratio R/R0
    axes[1].plot(t_s, R_um / (R0 * 1e6), color=colors[i], linewidth=2,
                 label=f"ΔC = {dC} mol/m³")

# ── Format left panel ────────────────────────────────────────────────
axes[0].set_xlabel("Time (s)", fontsize=12)
axes[0].set_ylabel("GUV Radius (µm)", fontsize=12)
axes[0].set_title("Radius over Time", fontsize=12)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(R0 * 1e6, color="gray", linestyle="--", alpha=0.5, label="Initial R₀")

# ── Format right panel ───────────────────────────────────────────────
axes[1].set_xlabel("Time (s)", fontsize=12)
axes[1].set_ylabel("Swelling Ratio  R / R₀", fontsize=12)
axes[1].set_title("Relative Swelling", fontsize=12)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="R₀ baseline")

plt.tight_layout()
plt.savefig("guv_osmotic_swelling.png", dpi=150, bbox_inches="tight")
plt.show()

print("Done! Figure saved as guv_osmotic_swelling.png")
print(f"\nPhysical setup:")
print(f"  Initial radius  : {R0*1e6:.1f} µm")
print(f"  Membrane Lp     : {Lp:.2e} m/Pa/s  (DOPC bilayer)")
print(f"  Temperature     : {T - 273.15:.1f} °C")
print(f"  Osmotic pressure range: {R_gas*T*delta_C_values[0]:.1f} – {R_gas*T*delta_C_values[-1]:.1f} Pa")