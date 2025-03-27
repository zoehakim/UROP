import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator, Aer
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator, Statevector
from scipy.optimize import curve_fit
from scipy.linalg import expm, fractional_matrix_power
from qiskit.compiler import transpile
import matplotlib.pyplot as plt
from functools import reduce
from qiskit_algorithms.eigensolvers import NumPyEigensolver


class DMFTroutine:
    def __init__(self, Urange, Vinit):
        self.Urange = Urange
        self.Vinit = Vinit
        self.max_iters=50
        self.tolerance=1e-6



    def gen_H(self, U, V):
        """Generate the Hamiltonian matrix H."""
        ham = SparsePauliOp.from_list([("ZIZI", U/4), ("XXII", V/2),("YYII",V/2),("IIXX",V/2),("IIYY",V/2)])
        return ham

    def get_U(self, t, H):
        """Compute the unitary evolution U(t) from the Hamiltonian H."""
        return expm(-1j * H * t)

    def find_GS(self, U, V):
        """Find the ground state and energy of the Hamiltonian H."""
        H = self.gen_H(U,V)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        ground_energy = eigenvalues[0]
        ground_state = eigenvectors[:, 0]
        return ground_energy, ground_state

    def get_GFr(self, t, H, gs):
        """Compute the Green's function at time t."""
        U = self.get_U(t, H)
        X1 = Pauli("IIIX").to_matrix()
        return (gs @ (X1 @ U.conj().T @ X1 @ U) @ gs).real

    def DMFT_step(self, U, V, tmax, plot=False):
        """Perform a single DMFT step, updating the value of V."""
        tvals = np.arange(0, tmax, tmax / 5000)
        g = np.zeros(np.shape(tvals), float)

        H = self.gen_H(U, V)
        psi = np.array([1] + [0] * 15, dtype=complex)
        en, gs = self.find_GS(U, V)

        # Calculate the Green's function for each time value
        for it, t in enumerate(tvals):
            g[it] = self.get_GFr(t, H, gs)

        # FFT to extract frequency components
        n = len(tvals)
        dt = tvals[1] - tvals[0]
        frequencies = np.fft.fftfreq(n, dt)
        window = np.hamming(n)
        fft_data = np.fft.fft(g * window)
        fft_magnitude = np.abs(fft_data)

        # Find the peaks in the FFT data
        mvals = [i for i in range(1, len(fft_data) - 1)
                if frequencies[i] > 0 and fft_magnitude[i - 1] < fft_magnitude[i] and fft_magnitude[i + 1] < fft_magnitude[i]]

        w1_est = 2 * np.pi * frequencies[mvals[0]]
        w2_est = 2 * np.pi * frequencies[mvals[1]]
        a_est = fft_magnitude[mvals[0]] / (fft_magnitude[mvals[0]] + fft_magnitude[mvals[1]])

        if plot:
            # Plot the time-domain signal and FFT
            plt.subplot(1, 2, 1)
            plt.plot(tvals, g, label='Time-domain signal')
            plt.plot(tvals, a_est * np.cos(w1_est * tvals) + (1 - a_est) * np.cos(w2_est * tvals), label='Fit')
            plt.xlabel('Time (t)')
            plt.ylabel('Amplitude')
            plt.title('Time-domain Signal')

            plt.subplot(1, 2, 2)
            plt.plot(frequencies[:n // 2], fft_magnitude[:n // 2], label='FFT Magnitude')
            for m in mvals:
                plt.axvline(frequencies[m], color="k")
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.title('FFT of the Signal')
            plt.tight_layout()
            plt.show()

        # Update V based on the computed Green's function frequencies
        Z = 1 / (V**4 * (a_est / (w1_est**4) + (1 - a_est) / (w2_est**4)))
        mixing_factor = 0.9
        V_new = mixing_factor * V + (1 - mixing_factor) * np.sqrt(Z)
        print(f"\t {V} -> {V_new}")
        return V_new

    def DMFT_run(self, U, V0, tmax):
        """Run the DMFT loop until convergence."""
        V = self.DMFT_step(U, V0, tmax)
        for i in range(100):
            V_new = self.DMFT_step(U, V, tmax)
            print(f"\t\t {i} {abs(V - V_new)}")
            if abs(V - V_new) < 2e-5:
                break
            V = V_new
        return V
    
    def findZ(self):
        Z = []
        V = self.Vinit
        for U in self.Urange:
            print(U)
            V = self.DMFT_run(U, V, 150)
            Z.append(V**2)

        return Z

