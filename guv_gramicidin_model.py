import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 100)

C_values = [0, 0.1, 0.5, 1, 5]

R0 = 10
k = 0.8

for C in C_values:
    R = R0 + k * C * t
    plt.plot(t, R, label=f"C = {C} µM")

plt.xlabel("Time")
plt.ylabel("GUV Radius (µm)")
plt.title("Effect of Gramicidin A on GUV Size")

plt.legend()
plt.show()