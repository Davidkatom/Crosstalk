import os
import random
from itertools import permutations
import pickle

import matplotlib.pyplot as plt
import numpy as np
import time

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator, noise
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from scipy.ndimage import gaussian_filter

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
import qiskit as qiskit
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from scipy.linalg import expm
from scipy.optimize import curve_fit, minimize, differential_evolution
from scipy.signal import find_peaks
from tqdm import tqdm

# Loading your IBM Quantum account(s)
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='4cb4dd43af6af16cc340156cf1dfebb32ec9f233cec56a0c6e98e9a574f99d96c7a4867d4541dc9394c58f81efe019ee4feb8fa17b475f79952b0f66bf000f4c'
)

h = lambda n, J, z: sum([J[i] * 0.25 * (z[i] - 1) * (z[(i + 1)] - 1) for i in range(n - 1)])

# prob_1 = 0.001  # 1-qubit gate
# prob_2 = 0.01  # 2-qubit gate
#
# T1 = [0,1,2,3,4]
# T2 = [0,1,2,3,4]
#
# # Depolarizing quantum errors
# error_1 = noise.depolarizing_error(prob_1, 1)
# error_2 = noise.depolarizing_error(prob_2, 2)
#
# # Add errors to noise model
# noise_model = noise.NoiseModel()
# noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
# noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
#
#
# # Get basis gates from noise model
# basis_gates = noise_model.basis_gates
# backend = AerSimulator(noise_model=noise_model,
#                        basis_gates=basis_gates)
service = QiskitRuntimeService()


# backend = service.backend("ibm_osaka")
# noise_model = NoiseModel.from_backend(backend)
# sim_noise = AerSimulator(noise_model=noise_model)
# passmanager = generate_preset_pass_manager(optimization_level=0, backend=sim_noise)
def effective_hem(size, J, W):
    hem = np.zeros((2 ** size, 2 ** size))
    for i in range(2 ** size):
        binary = '{0:b}'.format(i).zfill(size)
        Z = [(-1) ** int(i) for i in binary]
        # Z.reverse()
        hem[i, i] = h(size, J, Z)
        hem[i, i] += sum([W[k] for k in range(size) if binary[k] == '1'])
    return hem


class RamseyExperiment:
    def __init__(self, n, delay, shots, J, W, L,
                 backend=Aer.get_backend("qasm_simulator"), manual=False):
        self.delay = delay
        self.n = n
        self.J = J
        self.L = L
        # W.reverse()
        self.W = W[::-1]
        self.J = J[::-1]
        self.zi = []
        self.shots = shots
        self.backend = backend
        self.circuit = None
        self.zi = []
        self.result = None
        self.z = None
        self.noise_model = {}

        # self.create_circuit_crosstalk()
        # self.create_circuit_detuning()

    def add_decay(self):
        for i in range(self.n):
            self.zi[i] = self.zi[i] * np.exp(-self.L[i] * self.delay)

    def add_noise(self):
        bins = 2 ** self.n
        num_samples = self.shots//5
        samples = np.random.randint(low=0, high=bins, size=num_samples)
        bin_counts = [np.sum(samples == i) for i in range(bins)]
        noise_model = {}
        for i in range(len(bin_counts)):
            binary = '{0:b}'.format(i).zfill(self.n)
            noise_model[binary] = bin_counts[i]
        self.noise_model = noise_model
        self.z = self._get_z_exp()


    def create_circuit_crosstalk(self):
        q = QuantumRegister(self.n)
        c = ClassicalRegister(self.n)
        U = expm((-1j * self.delay) * effective_hem(self.n, self.J, self.W))
        U = qi.Operator(U)
        circuit = QuantumCircuit(q, c)

        for i in range(0, self.n, 2):
            circuit.h(i)
        for i in range(1, self.n, 4):
            circuit.x(i)

        circuit.barrier()
        circuit.unitary(U, [i for i in range(self.n)])
        circuit.barrier()
        for i in range(0, self.n, 2):
            circuit.h(i)

        for i in range(0, self.n - 1, 2):
            circuit.measure(i, c[4 * int(i / 4) + int((i / 2) % 2)])

        circuit.barrier()
        for i in range(self.n):
            circuit.reset(i)

        # Second half
        for i in range(0, self.n, 2):
            circuit.h(i)
        for i in range(3, self.n, 4):
            circuit.x(i)

        circuit.barrier()
        circuit.unitary(U, [i for i in range(self.n)])
        circuit.barrier()
        for i in range(0, self.n, 2):
            circuit.h(i)
        for i in range(2, self.n, 2):
            circuit.measure(i, c[2 * int(i / 4) + int((i / 2) % 2) + 1])
        self.circuit = circuit

        self.result = self._run()
        self.z = self._get_z_exp()

    def create_circuit_detuning(self):
        q = QuantumRegister(self.n)
        c = ClassicalRegister(self.n)
        U = expm((-1j * self.delay) * effective_hem(self.n, self.J, self.W))
        U = qi.Operator(U)
        circuit = QuantumCircuit(q, c)
        for i in range(0, self.n, 2):
            circuit.h(i)
        circuit.barrier()

        circuit.unitary(U, [i for i in range(self.n)])

        circuit.barrier()
        for i in range(0, self.n, 2):
            circuit.h(i)
            circuit.measure(i, c[i])
            circuit.reset(i)

        circuit.barrier()
        # # #### second half
        for i in range(1, self.n, 2):
            circuit.h(i)
        circuit.barrier()

        circuit.unitary(U, [i for i in range(self.n)])

        circuit.barrier()
        for i in range(1, self.n, 2):
            circuit.h(i)
            circuit.measure(i, c[i])
        circuit.barrier()
        circuit.draw(output='mpl')  # 'mpl' for matplotlib drawing

        self.circuit = circuit

        self.result = self._run()
        self.z = self._get_z_exp()

    def _run(self):
        # service = QiskitRuntimeService()
        # backend = service.backend("ibm_osaka")
        # noise_model = NoiseModel.from_backend(backend)

        # circ_tnoise = passmanager.run(self.circuit)
        # job = sim_noise.run(circ_tnoise)

        job = execute(self.circuit, Aer.get_backend("qasm_simulator"), shots=self.shots)
        return job.result()

    def _get_z_exp(self):
        Z = []
        Zi = []
        counts_exp = self.result.get_counts()
        counts = {key: counts_exp.get(key, 0) + self.noise_model.get(key, 0) for key in set(counts_exp) | set(self.noise_model)}
        normalization = sum(counts.values())
        for i in range(self.n):
            sumZ = 0
            for outcome, count in counts.items():
                outcome = outcome[::-1]
                plus = count if outcome[i] == '0' else 0
                minus = count if outcome[i] == '1' else 0
                sumZ += plus - minus
                Z.append((plus - minus) / normalization)
            Zi.append(sumZ / normalization)
        self.zi = Zi
        return sum(Z)


class RamseyBatch:

    def __init__(self, RamseyExperiments: "list of RamseyExperiment"):
        self.dist = None
        self.J = None
        self.W = None
        self.J_fit = []
        self.W_fit = []
        self.delay = []
        self.Z = []
        self.Zi = []
        self.n = None

        self.RamseyExperiments = RamseyExperiments
        for RamseyExperiment in RamseyExperiments:
            if self.J is None:
                self.W = RamseyExperiment.W
                self.J = RamseyExperiment.J
                self.n = RamseyExperiment.n
            self.delay.append(RamseyExperiment.delay)
            self.Z.append(RamseyExperiment.z)
            self.Zi.append(RamseyExperiment.zi)

    def get_zi(self, n):
        return [sublist[n] for sublist in self.Zi]
