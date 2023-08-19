import os
import random
from itertools import permutations
import pickle

import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.ndimage import gaussian_filter

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
import qiskit as qiskit
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from scipy.linalg import expm
from scipy.optimize import curve_fit

# Loading your IBM Quantum account(s)
IBMQ.save_account(
    '280beccbee94456a161a6cbc217e1366bc278bf60e22bd30281fa0ca5bec6e50897278ef818f3c53f6700e04b9ed32ea364195044413b7e02836a79d886b03d9',
    overwrite=True)
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-research')

h = lambda n, J, z: sum([J[i] * (z[i] - 1) * (z[(i + 1) % n] - 1) for i in range(n)])


def effective_hem(size, J):
    hem = np.zeros((2 ** size, 2 ** size))
    for i in range(2 ** size):
        binary = '{0:b}'.format(i).zfill(size)
        Z = [(-1) ** int(i) for i in binary]
        hem[i, i] = h(size, J, Z)
    return hem


class RamseyExperiment:
    '''
    A class to represent a Ramsey experiment.
    fields:
        n: number of qubits
        delay: how long to let the hamilotnian evolve
        shots: number of shots
        J: list of J values
        backend: backend to run the experiment on
    '''

    def __init__(self, n, delay, shots, J, name, backend=Aer.get_backend("qasm_simulator")):
        file_path = "experiments/" + name + ".pkl"
        self.name = name
        if os.path.exists(f'exp/{self.name}.pkl'):
            with open(f'exp/{self.name}.pkl', 'rb') as f:
                obj = pickle.load(f)
                self.__dict__.update(obj.__dict__)
        else:
            self.delay = delay
            self.n = n
            self.J = J
            self.backend = backend
            self.shots = shots
            self.circuit = self._create_circuit()
            self.result = self._run()
            self.z = self._get_z_exp()
            # with open(f'exp/{self.name}.pkl', 'wb') as f:
            #    pickle.dump(self, f)

    def _create_circuit(self):
        q = QuantumRegister(self.n)
        c = ClassicalRegister(self.n)
        circuit = QuantumCircuit(q, c)
        for i in range(self.n):
            circuit.h(i)
        circuit.barrier()
        U = expm((-1j * self.delay) * effective_hem(self.n, self.J))
        U = qi.Operator(U)

        circuit.unitary(U, [i for i in range(self.n)])

        circuit.barrier()

        for i in range(self.n):
            circuit.h(i)
        circuit.measure_all()
        return circuit

    def _run(self):
        job = execute(self.circuit, self.backend, shots=self.shots)
        return job.result()

    def _get_z_exp(self):
        Zi = []
        for i in range(self.n):
            for outcome, count in self.result.get_counts().items():
                plus = [count if outcome[i] == '0' else 0]
                minus = [count if outcome[i] == '1' else 0]
                Zi.append((sum(plus) - sum(minus)) / self.shots)

        return sum(Zi)  # TODO SHOULD I DIVIDE? / self.n


class RamseyBatch:

    def __init__(self, RamseyExperiments: "list of RamseyExperiment"):
        self.J = None
        self.J_fit = []
        self.delay = []
        self.Z = []
        self.n = None

        self.RamseyExperiments = RamseyExperiments
        for RamseyExperiment in RamseyExperiments:
            if self.J is None:
                self.J = RamseyExperiment.J
                self.n = RamseyExperiment.n
            self.delay.append(RamseyExperiment.delay)
            self.Z.append(RamseyExperiment.z)
        self.J_fit = self._calc_dist_using_fit()
        self.dist = self._calc_dist()

    def _calc_dist_using_fit(self):
        def func(t, *js):
            n = len(js)
            result = 2 * n
            for i in range(n):
                print(t)
                result += 4 * np.cos(4 * js[i] * t)
            for i in range(n):
                result += 2 * np.cos(4 * (js[i] + js[(i + 1) % n]) * t)
            return result / (2 ** (n - (n - 3)))

        initial_js = np.ones(self.n)
        try:

            # guess, pcov = curve_fit(func, self.delay, self.Z, p0=initial_js, bounds=(0, 2))

            from scipy.optimize import least_squares
            def residuals(params, t, y_obs):
                return func(t, *params) - y_obs

            bounds = ([0] * len(initial_js), [2] * len(initial_js))  # Bounds for the parameters
            initial_js = [1]*self.n  # Example initial guess

            guess = least_squares(residuals, initial_js, loss='huber', args=(self.delay, self.Z)).x

            # TODO check with theory
        except RuntimeError:
            print(f"Failed to converge. Skipping...")
            return list(initial_js)
        return guess

    def _calc_dist(self):
        min_dist = float('inf')
        for perm in permutations(self.J):
            dist = np.sum([np.abs(ai - bi) for ai, bi in zip(self.J_fit, perm)])
            if dist < min_dist:
                min_dist = dist
        return min_dist / (np.sqrt(self.n))
