import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from functools import reduce


# Define Pauli matrices and identity
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Function to create Pauli operators on specific qubits
def create_operator(op, qubit_pos, size=4):
    operators = [I] * size
    operators[qubit_pos] = op
    return reduce(np.kron, operators)


# Construct four-qubit operators
X_ops = [create_operator(X, i) for i in range(4)]
Y_ops = [create_operator(Y, i) for i in range(4)]
Z_ops = [create_operator(Z, i) for i in range(4)]

def print_state(state):
    """Print the quantum state in |0⟩/|1⟩ notation."""
    string = "|"
    for Z in Z_ops:
        val = state @ Z @ state
        string += "0" if abs(val - 1) < 1e-3 else "1"
    string += "⟩"
    print(string)

def gen_H(U, V):
    """Generate the Hamiltonian matrix H."""
    return U / 4 * (Z_ops[0] @ Z_ops[2]) + V / 2 * (X_ops[0] @ X_ops[1] + Y_ops[0] @ Y_ops[1] +
                                                    X_ops[2] @ X_ops[3] + Y_ops[2] @ Y_ops[3])

def get_U(t, H):
    """Compute the unitary evolution U(t) from the Hamiltonian H."""
    return expm(-1j * H * t)

def find_GS(H):
    """Find the ground state and energy of the Hamiltonian H."""
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    ground_energy = eigenvalues[0]
    ground_state = eigenvectors[:, 0]
    return ground_energy, ground_state

def get_GFr(t, H, gs):
    """Compute the Green's function at time t."""
    U = get_U(t, H)
    return (gs @ (X_ops[0] @ U.conj().T @ X_ops[0] @ U) @ gs).real

def DMFT_step(U, V, tmax, plot=False):
    """Perform a single DMFT step, updating the value of V."""
    tvals = np.arange(0, tmax, tmax / 5000)
    g = np.zeros(np.shape(tvals), float)

    H = gen_H(U, V)
    psi = np.array([1] + [0] * 15, dtype=complex)
    en, gs = find_GS(H)

    # Calculate the Green's function for each time value
    for it, t in enumerate(tvals):
        g[it] = get_GFr(t, H, gs)

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

def DMFT_run(U, V0, tmax):
    """Run the DMFT loop until convergence."""
    V = DMFT_step(U, V0, tmax)
    for i in range(100):
        V_new = DMFT_step(U, V, tmax)
        print(f"\t\t {i} {abs(V - V_new)}")
        if abs(V - V_new) < 2e-5:
            break
        V = V_new
    return V

if __name__ == "__main__":
    # Define U values and iterate through them
    Uvals = np.arange(0.5,6.0,0.5)
    Z = []
    V = 1.0
    for U in Uvals:
        print(U)
        V = DMFT_run(U, V, 150)
        Z.append(V**2)

    # Plot the results
    plt.figure()
    plt.plot(Uvals, Z)
    plt.xlabel('U')
    plt.ylabel('Z')
    plt.title('DMFT Results')
    plt.show()
