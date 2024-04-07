import Ramsey_ExperimentV2
import random
from scipy import signal
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumRegister, ClassicalRegister, execute
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, Aer
from scipy.linalg import expm
from scipy.optimize import curve_fit, minimize
from Ramsey_ExperimentV2 import effective_hem


class SpinEcho(Ramsey_ExperimentV2.RamseyExperiment):
    def __init__(self, n, delay, shots, J, W, L,
                 backend=Aer.get_backend("qasm_simulator"), basis="X"):
        super().__init__(n, delay, shots, J, W, L,
                         backend=Aer.get_backend("qasm_simulator"), basis="X")

    def create_echo_sequence(self, t, t0):
        q = QuantumRegister(self.n)
        c = ClassicalRegister(self.n)
        U0 = expm((-1j * t0) * effective_hem(self.n, self.J, self.W))
        U = expm((-1j * t) * effective_hem(self.n, self.J, self.W))

        U = qi.Operator(U)
        circuit = QuantumCircuit(q, c)

        for i in range(self.n):
            circuit.h(i)
        circuit.barrier()
        circuit.unitary(U0, [i for i in range(self.n)])
        circuit.barrier()

        for i in range(self.n):
            circuit.y(i)

        circuit.barrier()

        circuit.unitary(U, [i for i in range(self.n)])
        circuit.barrier()
        for i in range(self.n):
            if self.basis == "Y":
                circuit.sdg(i)
            circuit.h(i)
            circuit.measure(i, c[i])
        self.circuit = circuit

        self.result = self._run()
        self.z = self._get_z_exp()
