import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.special
#from scipy.optimize import fsolve
#from scipy.integrate import solve_ivp
#from scipy.linalg import eigh_tridiagonal



### POTENTIAL WELL ###

# Constants
L = 1  # Length of the potential well
hbar = 1  # Reduced Planck's constant for simplicity
m = 1  # Mass for simplicity

# Define the eigenstates for the infinite potential well
def infinite_potential_well_wavefunction(x, n):
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

# X values
x = np.linspace(0, L, 400)
x2 = np.linspace(-2*L, 2*L, 4000)

# Plotting
plt.figure()

# Potential Well
plt.fill_between(x, -5, 0, where=(x <= L), color='lightblue', label='Potential Well')
plt.fill_between(x2, -5, 50, where=(x2 <= 0), color='lightblue')
plt.fill_between(x2, -5, 50, where=(x2 >= L), color='lightblue')

# Plotting eigenstates with energy displacement
for n in range(1, 6):
    energy = n**2  # Simple energy level scaling
    plt.plot(x, infinite_potential_well_wavefunction(x, n) + energy, label=f'n={n}, E={energy}')
    plt.plot([x2[0], x2[-1]], [energy, energy], "k--")

plt.title("Eigenstates of the Infinite Potential Well")
plt.xlim(-0.15*L, 1.15*L)
plt.ylim(-2, 30)
plt.xlabel("Position (x)")
plt.ylabel("Wavefunction (ψ) + Energy Level")
plt.legend()
plt.tight_layout()
plt.show()








### HYDROGEN ATOM ###

# Constants
a_0 = 1  # Scaled Bohr radius for simplicity

# Define the radial wavefunctions for hydrogen
def hydrogen_wavefunction(r, n):
    r = np.abs(r)
    if n == 1:
        return (1 / np.sqrt(np.pi * a_0**3)) * np.exp(-r / a_0)
    elif n == 2:
        return (1 / (2 * np.sqrt(2 * np.pi * a_0**3))) * (1 - r / (2 * a_0)) * np.exp(-r / (2 * a_0))
    elif n == 3:
        return (1 / (81 * np.sqrt(3 * np.pi * a_0**3))) * (27 - 18*r/a_0 + 2*r**2/a_0**2) * np.exp(-r / (3 * a_0))
    else:
        return np.zeros_like(r)

# X values (r values)
r = np.linspace(-25, 25, 1000)

# Energies for n = 1 to 5
energies = [-13.6 / n**2 for n in range(1, 6)]

# Plotting
plt.figure()
plt.fill_between(r, -15, -20/np.abs(r), color='lightblue', label='Potential') # prefactor of 25 is only there to make the plot look nice

# Plotting the wavefunctions with energy displacement
for n in range(1, 4):
    plt.plot(r, hydrogen_wavefunction(r, n)/hydrogen_wavefunction(0, n) + energies[n-1], label=f'n={n}, E={energies[n-1]:.2f} eV')
    plt.plot(r, [energies[n-1] for r_ in r], "k--")

# Setup the plot
plt.title("Radial Wavefunctions of the Hydrogen Atom (First 3 States)")
plt.ylim([-15, 1])
plt.xlabel("Radius (r)")
plt.ylabel("Wavefunction (ψ) + Energy Level")
plt.legend()
plt.tight_layout()
plt.show()

### Also check out https://www.falstad.com/qmatom/ for better visualization ###








### MORSE POTENTIAL ###

# Constants
D = 10
r_e = 1
a = 1
m = 1
e = 1
nu_0 = a/(2*np.pi) * np.sqrt( 2*D*e/m )
lamb = np.sqrt(2*m*D/e)/a

# Define potential
def V_0(x):	# [eV]
		return D * ( np.exp(-a*(x-r_e)) -1. )**2 

# Calculate number of bound states
N = (2*D-2*np.pi*nu_0)/(2*np.pi*nu_0)
print("number of bound states: ", N)

# Define the wavefunctions and energies
def analytic_eingenenergies_Morse(n):
  return 2*np.pi*hbar * nu_0 * (n+0.5) - (2*np.pi)**2*hbar**2*nu_0**2/(4*D)*(n+0.5)**2 - 0.045715676976908599

def analytic_eingenfuncs_Morse(r, n):
  x     = a * r
  x_e   = a * r_e
  Gamma = scipy.special.gamma(2*lamb-n)
  N_n   = np.sqrt( math.factorial(n) * (2*lamb-2*n-1) / (Gamma) ) 
  z     = 2 * lamb * np.exp(-(x-x_e))
  L_n   = scipy.special.genlaguerre(n, 2*lamb-2*n-1)
  return N_n * z**(lamb-n-0.5) * np.exp(-0.5*z) * L_n(z)
  # Notice that this function is for some reason not(!) normalized .... the equation was taken from wikipedia

# Setup the plot
r = np.linspace(0, 10, 1000)
plt.figure()
plt.fill_between(r, -1, V_0(r), color='lightblue', label='Potential')
for n in range(int(N)):
  plt.plot(r, analytic_eingenfuncs_Morse(r, n) + analytic_eingenenergies_Morse(n))
  plt.plot([r[0], r[-1]], [analytic_eingenenergies_Morse(n), analytic_eingenenergies_Morse(n)], "k--")

plt.title("Wavefunctions of the Morse Potential")
plt.xlabel("x")
plt.ylabel("Wavefunction (ψ) + Energy Level")
plt.legend()
plt.tight_layout()

plt.show()










