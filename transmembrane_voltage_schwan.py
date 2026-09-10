"""
Transmembrane Voltage During Electroporation — Schwan Equation
===============================================================
When an external electric field is applied to a GUV or cell,
the transmembrane voltage (TMV) builds up over time.

The Schwan equation describes the steady-state TMV on a spherical
membrane in an external field:

    V_tm = 1.5 * R * E * cos(theta)

For the time-dependent buildup (RC charging model):

    V_tm(t) = 1.5 * R * E * cos(theta) * (1 - exp(-t / tau))

    where:
      R     = cell/GUV radius (m)
      E     = external electric field (V/m)
      theta = polar angle (0 = field-facing pole)
      tau   = membrane charging time constant (s)
              tau = R * Cm / (2 * sigma_m)  [simplified]

Electroporation threshold: V_tm ~ 0.2 – 1.0 V

Author: [Tamanna Mostafa]
date:2026-04-21

Reference: Schwan, H.P. (1957). Electrical properties of tissue.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Vesicle / cell parameters ────────────────────────────────────────
R = 10e-6         # m       GUV radius (10 µm)
Cm = 1e-2          # F/m²    specific membrane capacitance (~1 µF/cm²)
sigma_i = 0.5       # S/m     intracellular conductivity

# ── Membrane charging time constant ─────────────────────────────────
# tau = R * Cm / (2 * sigma_i)  — simplified single-shell model
tau = R * Cm / (2 * sigma_i)
print(f"Membrane charging time constant τ = {tau*1e6:.2f} µs")

# ── Electric field strengths to compare ─────────────────────────────
E_values = [100, 500, 1000, 2000, 5000]   # V/m

# ── Electroporation threshold ────────────────────────────────────────
V_threshold = 0.5   # V  (typical literature value for GUVs)

# ── Time array ───────────────────────────────────────────────────────
t = np.linspace(0, 10 * tau, 300)         # simulate 10× time constants

# ── Polar angle: theta = 0 (pole facing field, maximum TMV) ─────────
theta = 0
cos_theta = np.cos(theta)                 # = 1.0 at the pole

# ── Plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Transmembrane Voltage Buildup — Schwan Equation",
             fontsize=14, fontweight="bold")

colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(E_values)))

for i, E in enumerate(E_values):
    # Steady-state TMV at the pole (theta = 0)
    V_ss = 1.5 * R * E * cos_theta

    # Time-dependent TMV (RC charging)
    V_tm = V_ss * (1 - np.exp(-t / tau))

    # Convert time to microseconds for readability
    t_us = t * 1e6

    axes[0].plot(t_us, V_tm, color=colors[i], linewidth=2,
                 label=f"E = {E} V/m  (V_ss = {V_ss*1000:.1f} mV)")

    # Bar chart: steady-state TMV per field strength
    axes[1].bar(i, V_ss * 1000, color=colors[i], alpha=0.85,
                label=f"{E} V/m")

# ── Threshold line ───────────────────────────────────────────────────
axes[0].axhline(V_threshold, color="red", linestyle="--", linewidth=1.5,
                label=f"Electroporation threshold ({V_threshold} V)")
axes[1].axhline(V_threshold * 1000, color="red", linestyle="--",
                linewidth=1.5, label="Threshold (500 mV)")

# ── Format left panel: V_tm vs time ─────────────────────────────────
axes[0].set_xlabel("Time (µs)", fontsize=12)
axes[0].set_ylabel("Transmembrane Voltage (V)", fontsize=12)
axes[0].set_title("TMV Buildup over Time (at θ = 0°)", fontsize=12)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=0)

# ── Format right panel: steady-state comparison ──────────────────────
axes[1].set_xticks(range(len(E_values)))
axes[1].set_xticklabels([f"{E}" for E in E_values], fontsize=10)
axes[1].set_xlabel("Electric Field Strength (V/m)", fontsize=12)
axes[1].set_ylabel("Steady-State TMV (mV)", fontsize=12)
axes[1].set_title("Steady-State TMV per Field Strength", fontsize=12)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("transmembrane_voltage_schwan.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSummary:")
print(f"  GUV radius         : {R*1e6:.0f} µm")
print(f"  Membrane Cm        : {Cm*1e2:.1f} µF/cm²")
print(f"  Charging time τ    : {tau*1e6:.2f} µs")
print(f"  Threshold voltage  : {V_threshold*1000:.0f} mV")
for E in E_values:
    V_ss = 1.5 * R * E
    status = ">> ELECTROPORATION" if V_ss >= V_threshold else "   sub-threshold"
    print(f"  E = {E:5d} V/m  →  V_tm = {V_ss*1000:6.1f} mV  {status}")
print("\nFigure saved as transmembrane_voltage_schwan.png")
