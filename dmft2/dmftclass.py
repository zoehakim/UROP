import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator, Aer
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator, Statevector
from scipy.optimize import curve_fit
from scipy.linalg import expm, fractional_matrix_power
from qiskit.compiler import transpile

class DMFTroutine:
    def __init__(self, Urange, Vinit):
        self.Urange = Urange
        self.Vinit = Vinit
        self.max_iters=50
        self.tolerance=1e-6

    def get_siam_hamiltonian(self,U,V):
        # Define the Hamiltonian using tensor products of Pauli operators
        H = SparsePauliOp.from_list([("ZIZI", U/4), ("XXII", V/2),("YYII",V/2),("IIXX",V/2),("IIYY",V/2)])
        return H.to_matrix()
    
    def trotter_step(self, H, dt):
        return expm(-1j * dt * H)  # Trotter approx.

    def greens_function(self, U, V, times, num_trotter_steps=20):
        H_matrix = self.get_siam_hamiltonian(U,V)
        #print(H_matrix)
        dt = times[1] - times[0]  # Small time step for trotterization
        U_t = self.trotter_step(H_matrix, dt / num_trotter_steps)  # Approximate evolution
    

        simulator = AerSimulator()
        greens_vals = []


        qc = QuantumCircuit(4)

        qc.h(0)
        initial_state = Statevector.from_instruction(qc).data.reshape(-1, 1)
        #print("initial_state", initial_state)
        
        X1 = Pauli("IIIX").to_matrix()
        #print('X1', X1)

        for t in times:
            evolved_state = np.linalg.matrix_power(U_t, int(t / dt)) @ initial_state
    #         print("u conj: ", U_t.conj())
    #         print("evolved state", evolved_state)
            expectation_value = np.real(np.conj(evolved_state.T) @ X1 @ evolved_state)
            #print('expectation value', expctation_value)
            greens_vals.append(expectation_value)
        #print('greens vals', greens_vals)
        return greens_vals

    def fit_greens_function(self, times, greens_values):
        greens_values = np.array(greens_values, dtype=float).flatten()
        times = np.array(times, dtype=float).flatten()
        greens_values = np.nan_to_num(greens_values, nan=0.0, posinf=0.0, neginf=0.0)

        def model(t, a, w, m):
            return a * np.cos(w * t) + (1 - a) * np.cos(m * t)

        params, _ = curve_fit(model, times, greens_values)
        return params  # [a, w, m]

    def calculate_quasiparticle_weight(self, V, a, w, m):
        denom = V**4 * ((a / w**4) + ((1 - a) / m**4))
        if denom == 0:
            denom = 1e-6  # Avoid division by zero
        return 1 / denom

    def self_consistency(self, V, Z):
        return np.abs(V**2 - Z) <= self.tolerance

    def dmft_routine(self):
        times = np.linspace(0, 10, 50)
        Z_values = []

        for U in self.Urange:
            V = self.Vinit
            for _ in range(self.max_iters): 
                greens_vals = self.greens_function(U, V, times)
                a, w, m = self.fit_greens_function(times, greens_vals)
                Z = self.calculate_quasiparticle_weight(V, a, w, m)

                if self.self_consistency(V, Z):
                    break 
                V = np.sqrt(Z)
            Z_values.append(Z)

        return Z_values
