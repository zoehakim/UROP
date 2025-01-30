import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator, Aer
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator, Statevector
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.linalg import expm, fractional_matrix_power
from qiskit.compiler import transpile
from dmftclass import DMFTroutine

# Define parameters
U_range = np.linspace(1,10,10) # Range of U values
V_init = 10.0  # Initial guess for V

dmft = DMFTroutine(U_range,V_init)

Z_values = dmft.getZ()

plt.plot(U_range, Z_values, marker="o")
plt.xlabel("Impurity On-Site Interaction Energy (U)")
plt.ylabel("Quasiparticle Weight (Z)")
plt.title("Quasiparticle Weight vs U")
plt.grid()
plt.show()