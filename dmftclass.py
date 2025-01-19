from qiskit import *
from qiskit_aer.aerprovider import AerSimulator
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram, plot_state_qsphere, circuit_drawer, plot_state_city, plot_bloch_multivector, plot_state_paulivec, plot_state_hinton, array_to_latex
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import TwoLocal

class Qisclass:
    def __init__(self, U, V):
        self.U = U
        self.V = V

    def greencirc(self):
        work = QuantumRegister(4)
        subcirc = QuantumCircuit(work)

        ansatz = TwoLocal(4, rotation_blocks='ry', entanglement_blocks='cz', reps=2)

        subcirc.append(ansatz, work)

        circAnc = QuantumCircuit(5,1)

        circ = circAnc.compose(subcirc, qubits=[1,4])

        circ.h(0)
        circ.x(0)
        #tensor
        circ.x(0)
        #timeev
        circ.h(0)
        circ.meas(0,0)


