import math
import numpy as np
import matplotlib.pyplot as plt

m = 1
omega = 1


def analytic_eingenfuncs(x, n):
		def H(n,x):
			if n == 0:
				return 1.
			elif n == 1:
				return 2*x
			elif n == 2:
				return 4*x**2 - 2
			elif n == 3:
				return 8*x**3 - 12*x
			elif n == 4:
				return 16*x**4 - 48*x**2 + 12
			else:
				print("error. this order hermite polynomial has not yet been implemented")
		i  = m*omega
		return 0.5 * 1./np.sqrt(2**n*math.factorial(n)) * H(n, x*np.sqrt(i)) * np.exp( -0.5*i*x**2 ) + omega*(n+0.5)  # notice that the energy for plotting was already added ....


x = np.linspace(-7, 7, 1000)

plt.figure()
plt.fill_between(x, -1, 0.5*omega*x**2, color='lightblue', label='Potential')
for n in range(5):
  plt.plot(x, analytic_eingenfuncs(x, n))
  plt.plot([x[0], x[-1]], [omega*(n+0.5), omega*(n+0.5)], "k--")

plt.ylim([-1, 6])
plt.title("Wavefunctions of the Harmonic Oscillator")
plt.xlabel("x")
plt.ylabel("Wavefunction (ψ) + Energy Level")
plt.legend()
plt.tight_layout()

plt.show()
