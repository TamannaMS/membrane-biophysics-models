import numpy as np
import matplotlib.pyplot as plt

def calculate_membrane_energy(voltage_range, thickness_nm=5, area_um2=1, epsilon_r=2.0):
    """
    Calculates the energy stored in a lipid bilayer (capacitor model).
    
    Parameters:
    voltage_range: Array of voltages (V)
    thickness_nm: Membrane thickness (typical ~5nm)
    area_um2: Membrane surface area (um^2)
    epsilon_r: Relative permittivity of lipid tail region (typically 2.0)
    """
    # Constants
    epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)
    
    # Unit conversions
    d = thickness_nm * 1e-9 # nm to m
    A = area_um2 * 1e-12    # um^2 to m^2
    
    # Calculate Capacitance (C = epsilon_0 * epsilon_r * A / d)
    capacitance = (epsilon_0 * epsilon_r * A) / d
    
    # Calculate Energy (E = 0.5 * C * V^2)
    energy = 0.5 * capacitance * (voltage_range**2)
    
    return capacitance, energy

# Simulation Parameters
voltages = np.linspace(0, 1.2, 100) # Voltage range from 0 to 1.2V (Electroporation threshold ~1V)
C, E = calculate_membrane_energy(voltages)

# Visualization
plt.figure(figsize=(8, 5))
plt.plot(voltages, E * 1e15, color='teal', linewidth=2) # Energy in femtojoules (fJ)
plt.axvline(x=1.0, color='red', linestyle='--', label='Electroporation Threshold (~1V)')
plt.title('Energy Storage in a Lipid Bilayer vs. Applied Voltage')
plt.xlabel('Transmembrane Potential (Volts)')
plt.ylabel('Stored Energy (fJ)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"Calculated Membrane Capacitance: {C*1e12:.4f} pF")