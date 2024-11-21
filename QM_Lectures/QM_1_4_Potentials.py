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
dt = 0.001


# Gaussian wave packet at time 0
def psi(x):
    psi = np.zeros_like(x, dtype=np.complex128)
    p_values = np.linspace(-10, 10, 1000)  # Range of p-values
    for p in p_values:
        c_p = np.exp(-0.5 * (p - p0)**2 / sigma_p**2)
        psi += c_p * np.exp(1j * (p * x))
    return psi


# Normalize wave packet
norm = np.sqrt(np.sum(np.abs(psi(x))**2) * dx)
psi_t = psi(x)/norm


# Second derivative operator (Laplace operator) for the Hamiltonian
def laplacian(psi, dx):
    return (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / dx**2


# Potential (free particle)
V = np.zeros_like(x)
for i in range(N):
  if x[i] > 15 and x[i] < 20:
    V[i] = 20


# RK4 method time evolution
def time_step_rk4(psi, dt, dx):
    def dpsi_dt(psi):
        laplacian_psi = laplacian(psi, dx)
        return -1j* ((-1 / (2 * m)) * laplacian_psi  + V * psi)
    k1 = dpsi_dt(psi)
    k2 = dpsi_dt(psi + 0.5 * dt * k1)
    k3 = dpsi_dt(psi + 0.5 * dt * k2)
    k4 = dpsi_dt(psi + dt * k3)
    psi_next = psi + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    psi_next[0:2] *= 0
    psi_next[-3:-1] *= 0
    return psi_next


# Set up the plot
fig, ax = plt.subplots()
ax.plot(x, V/max(V)*0.5, "k-", label="potential")
line_r, = ax.plot(x, np.real(psi_t), label="real")
line_i, = ax.plot(x, np.imag(psi_t), label="imag")
line_probab, = ax.plot(x, np.abs(psi_t), lw=2, label="probability")

plt.legend()
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, 1)
ax.set_xlabel('x')
ax.set_ylabel('Psi(x, t)')


# Update function for animation
def update(frame):
    global psi_t
    for i in range(50):
      psi_t = time_step_rk4(psi_t, dt, dx)
    line_r.set_ydata(np.real(psi_t))
    line_i.set_ydata(np.imag(psi_t))
    line_probab.set_ydata(np.abs(psi_t))
    return line_r, line_i, line_probab


# Create the animation
ani = FuncAnimation(fig, update, frames=50, interval=50)
plt.show()


