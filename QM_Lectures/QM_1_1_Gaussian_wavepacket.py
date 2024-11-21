import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0     # Particle mass
p0 = 5.0    # Central wavenumber
sigma_p = 1.0  # Width of Gaussian in momentum-space


# Discretization
x_min, x_max = -10, 25
dx = 0.05
x = np.arange(x_min, x_max, dx)
N = len(x)


# Gaussian wave packet as a function of time
def psi_t(x, t):
    psi = np.zeros_like(x, dtype=np.complex128)
    p_values = np.linspace(-10, 10, 1000)  # Range of p-values
    for p in p_values:
        E_p = hbar * p**2 / (2 * m)  # Dispersion relation
        c_p = np.exp(-0.5 * (p - p0)**2 / sigma_p**2)
        psi += c_p * np.exp(1j * (p * x - E_p * t))
    return psi


# Normalize wave packet
norm = np.sqrt(np.sum(np.abs(psi_t(x, 0))**2) * dx)


# Set up the plot
fig, ax = plt.subplots()
line_r, = ax.plot(x, np.real(psi_t(x, 0)/norm), label="real")
line_i, = ax.plot(x, np.imag(psi_t(x, 0)/norm), label="imag")
line_probab, = ax.plot(x, np.abs(psi_t(x, 0)/norm), lw=2, label="probability")

plt.legend()
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, 1)
ax.set_xlabel('x')
ax.set_ylabel('Psi(x, t)')


# Update function for animation
def update(frame):
    t = frame * 0.1
    psi = psi_t(x, t) / norm
    line_r.set_ydata(np.real(psi))
    line_i.set_ydata(np.imag(psi))
    line_probab.set_ydata(np.abs(psi))
    return line_r, line_i, line_probab


# Create the animation
ani = FuncAnimation(fig, update, frames=50, interval=50)
plt.show()



