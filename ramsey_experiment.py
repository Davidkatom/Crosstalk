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
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks

# Loading your IBM Quantum account(s)
IBMQ.save_account(
    '280beccbee94456a161a6cbc217e1366bc278bf60e22bd30281fa0ca5bec6e50897278ef818f3c53f6700e04b9ed32ea364195044413b7e02836a79d886b03d9',
    overwrite=True)
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-research')

h = lambda n, J, z: sum([J[i] * (z[i] - 1) * (z[(i + 1) % n] - 1) for i in range(n)])


def func(t, *js):
    n = len(js)
    result = 2 * n
    for i in range(n):
        result += 4 * np.cos(4 * js[i] * t)
    for i in range(n):
        result += 2 * np.cos(4 * (js[i] + js[(i + 1) % n]) * t)
    return result / (2 ** (n - (n - 3)))


def grad(t, *js):
    g = []
    for i in range(len(js)):
        g.append(-2 * t * np.sin(4 * js[i] * t) - t * np.sin(4 * (js[i] + js[(i + 1) % len(js)]) * t) - t * np.sin(
            4 * (js[i] + js[(i - 1) % len(js)]) * t))
    return np.array(g)


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

    def __init__(self, n, delay, shots, J, name="0", backend=Aer.get_backend("qasm_simulator"), manual=False):
        file_path = "experiments/" + name + ".pkl"
        self.name = name
        if os.path.exists(f'exp/{self.name}.pkl') and name != "0":
            with open(f'exp/{self.name}.pkl', 'rb') as f:
                obj = pickle.load(f)
                self.__dict__.update(obj.__dict__)
        else:
            self.delay = delay
            self.n = n
            self.J = J
            self.zi = []
            self.shots = shots
            if not manual:
                self.backend = backend
                self.circuit = self._create_circuit()
                self.result = self._run()
                self.z = self._get_z_exp()
            # with open(f'exp/{self.name}.pkl', 'wb') as f:
            #    pickle.dump(self, f)
            else:
                self.z = 0
                self._manuel_run()

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
        Z = []
        Zi = []
        for i in range(self.n):
            sumZ = 0
            for outcome, count in self.result.get_counts().items():
                plus = [count if outcome[i] == '0' else 0]
                minus = [count if outcome[i] == '1' else 0]
                sumZ += sum(plus) - sum(minus)
                Z.append((sum(plus) - sum(minus)) / self.shots)
            Zi.append(sumZ / self.shots)
        self.zi = Zi
        return sum(Z)  # TODO SHOULD I DIVIDE? / self.n

    def _manuel_run(self):
        Zi = []
        for i in range(self.n):
            zi = 1 / 4
            zi = zi * (1 + np.cos(4 * self.J[i] * self.delay) + np.cos(
                4 * (self.J[i] + self.J[(i - 1) % self.n]) * self.delay) + np.cos(
                4 * self.J[(i - 1) % self.n] * self.delay))
            Zi.append(zi)
        self.z = sum(Zi)
        self.zi = Zi


class RamseyBatch:

    def __init__(self, RamseyExperiments: "list of RamseyExperiment"):
        self.dist = None
        self.J = None
        self.J_fit = []
        self.delay = []
        self.Z = []
        self.Zi = []
        self.n = None

        self.RamseyExperiments = RamseyExperiments
        for RamseyExperiment in RamseyExperiments:
            if self.J is None:
                self.J = RamseyExperiment.J
                self.n = RamseyExperiment.n
            self.delay.append(RamseyExperiment.delay)
            self.Z.append(RamseyExperiment.z)
            self.Zi.append(RamseyExperiment.zi)
        # if method == "curve_fit":
        #     self.J_fit = self._curve_fit()
        # if method == "least_squares":
        #     self.J_fit = self._least_squares()

    def full_fft(self):
        initial_guess = []
        extended = self.Z[::-1]
        extended = extended + self.Z
        fft_output_ext = np.fft.fft(extended)
        sample_rate = len(self.delay) / self.delay[-1]  # Sampling rate of your data (change if known)
        frequencies_ext = np.fft.fftfreq(2 * len(self.Z), 1 / sample_rate)

        positive_indices = np.where(frequencies_ext > 0)
        positive_magnitudes = np.abs(fft_output_ext)[positive_indices]
        peaks, _ = find_peaks(positive_magnitudes)

        # Get the magnitudes of these peaks
        peak_magnitudes = positive_magnitudes[peaks]

        # Sort the peaks by their magnitudes in descending order
        sorted_peak_indices = np.argsort(peak_magnitudes)[::-1]

        # Choose the n highest peaks
        n_highest_peaks = sorted_peak_indices[:self.n]

        for peak_index in n_highest_peaks:
            initial_guess.append(frequencies_ext[positive_indices][peaks[peak_index]] * (0.5 * np.pi))
        initial_guess = initial_guess
        print("Initial guess from fft: ", initial_guess)
        return initial_guess

    def fft(self):
        def extract_two_closest_to_zero(frequencies, peaks):
            """Extract the two peaks closest to zero from the given peaks."""
            # Sort peaks based on their absolute distance to zero
            sorted_peaks = sorted(peaks, key=lambda x: abs(frequencies[x]))
            peaks = sorted_peaks[:2]
            if peak_magnitudes[peaks[0]] / peak_magnitudes[peaks[1]] > 5:
                peaks[1] = peaks[0]
            elif peak_magnitudes[peaks[1]] / peak_magnitudes[peaks[0]] > 5:
                peaks[0] = peaks[1]
            return peaks

        sample_rate = len(self.delay) / self.delay[-1]  # Sampling rate of your data (change if known)
        peak_pairs = []
        for i in range(self.n):
            extended = self.get_zi(i)[::-1]
            extended = extended + self.get_zi(i)
            fft_output_ext = np.fft.fft(extended)
            frequencies_ext = np.fft.fftfreq(2 * len(self.get_zi(i)), 1 / sample_rate)

            positive_indices = np.where(frequencies_ext > 0)
            positive_magnitudes = np.abs(fft_output_ext)[positive_indices]

            # Find peaks in the positive magnitudes
            peaks, _ = find_peaks(positive_magnitudes)

            # Get the magnitudes of these peaks
            peak_magnitudes = positive_magnitudes[peaks]

            # Sort the peaks by their magnitudes in descending order
            sorted_peak_indices = np.argsort(peak_magnitudes)[::-1]
            n_highest_peaks = sorted_peak_indices[:3]

            # Extract two peaks closest to zero
            selected_peaks = extract_two_closest_to_zero(frequencies_ext[positive_indices], n_highest_peaks)

            freq = []
            for peak_index in selected_peaks:
                freq.append(frequencies_ext[positive_indices][peaks[peak_index]] * (0.5 * np.pi))

            peak_pairs.append((freq[0], freq[1]))

        def find_ordered_js(peak_pairs):
            ordered_js = []
            n = len(peak_pairs)

            # Start with the first qubit
            current_peaks = list(peak_pairs[0])

            for i in range(n):
                next_peaks = peak_pairs[(i + 1) % n]
                # Find the closest peak between the two sets
                distances = (min([abs(current_peaks[0] - p) for p in next_peaks]),
                             min([abs(current_peaks[1] - p) for p in next_peaks]))

                if (distances[0] < distances[1]):
                    ordered_js.append(current_peaks[0])
                else:
                    ordered_js.append(current_peaks[1])

                current_peaks = next_peaks

            return ordered_js

        ordered_js = find_ordered_js(peak_pairs)
        # self.dist = self._calc_dist()
        return ordered_js

    def curve_fit_grad(self, use_fft=False):
        def objective_function(params, x, y, func, grad_func):
            # Compute the predicted values
            predicted_y = func(x, *params)

            # Compute and return the sum of squared residuals
            residuals = y - predicted_y
            return np.sum(residuals ** 2)

        def gradient_objective_function(params, x, y, func, grad_func):

            # Compute the residuals
            residuals = y - func(x, *params)

            # Compute the gradient of the model function
            grad_f = grad_func(x, *params)

            # Compute the gradient of the objective function
            grad_objective = -2 * np.dot(residuals, grad_f)

            return grad_objective
        if use_fft:
            initial_js = self.fft()
            print("Finished FFT")
        else:
            initial_js = np.ones(self.n)

        result = minimize(
            objective_function, initial_js,
            jac=gradient_objective_function,
            args=(self.delay, self.Z, func, grad),
            method='L-BFGS-B'
        )
        self.J_fit = result.x

    def curve_fit(self, use_fft=False):
        if use_fft:
            initial_js = self.fft()
            print("Finished FFT")
        else:
            initial_js = np.ones(self.n)

        try:
            guess, pcov = curve_fit(func, self.delay, self.Z, p0=initial_js,
                                    bounds=(min(initial_js) - 1, max(initial_js) + 1), method='dogbox')
            self.J_fit = guess
        except RuntimeError:
            print(f"Failed to converge. Skipping...")
            self.J_fit = list(initial_js)
        self.dist = self._calc_dist()
        print("Finished curve_fit")

    def least_squares(self):

        initial_js = np.ones(self.n)
        try:
            from scipy.optimize import least_squares
            def residuals(params, t, y_obs):
                res = []
                for delay, z_obs in zip(t, y_obs):
                    res.append(func(delay, *params) - z_obs)
                return res

            bounds = ([0] * len(initial_js), [2] * len(initial_js))  # Bounds for the parameters

            initial_js = [1] * self.n  # Example initial guess

            guess = least_squares(residuals, initial_js, bounds=bounds, args=(self.delay, self.Z)).x

            # TODO check with theory
        except RuntimeError:
            print(f"Failed to converge. Skipping...")
            return list(initial_js)
        return guess

    def _calc_dist(self):
        distance = np.sqrt(np.sum((self.J - self.J_fit) ** 2))
        return distance / (np.sqrt(self.n))

    def get_zi(self, n):
        return [sublist[n] for sublist in self.Zi]
