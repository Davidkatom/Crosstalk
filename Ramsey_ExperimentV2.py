import random
from scipy import signal
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumRegister, ClassicalRegister, execute
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, Aer
from scipy.linalg import expm
from scipy.optimize import curve_fit, minimize

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
                 backend=Aer.get_backend("qasm_simulator"), basis="X"):
        self.delay = delay
        self.n = n
        self.J = J
        self.L = L[::-1]
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
        self.raw_data = []
        self.qubits_measured = []
        self.basis = basis

    def create_full_circuit(self):
        q = QuantumRegister(self.n)
        c = ClassicalRegister(self.n)
        U = expm((-1j * self.delay) * effective_hem(self.n, self.J, self.W))
        U = qi.Operator(U)
        circuit = QuantumCircuit(q, c)

        for i in range(self.n):
            circuit.h(i)
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

    def add_decay(self):
        for i in range(self.n):
            self.zi[i] = self.zi[i] * np.exp(-self.L[self.n - i - 1] * self.delay)

    def add_noise(self):
        bins = 2 ** self.n
        num_samples = self.shots // 5
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
            if self.basis == "Y":
                circuit.sdg(i)
            circuit.h(i)

        for i in range(0, self.n, 2):
            circuit.measure(i, c[4 * int(i / 4) + int((i / 2) % 2)])
            self.qubits_measured.append(i)

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
            if self.basis == "Y":
                circuit.sdg(i)
            circuit.h(i)
        for i in range(2, self.n, 2):
            circuit.measure(i, c[2 * int(i / 4) + int((i / 2) % 2) + 1])
            self.qubits_measured.append(i)
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
            if self.basis == "Y":
                circuit.sdg(i)
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
            if self.basis == "Y":
                circuit.sdg(i)
            circuit.h(i)
            circuit.measure(i, c[i])
        circuit.barrier()

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
        # self.raw_data = job.result().get_memory()
        self.raw_data = self.get_raw_from_counts(job.result().get_counts())
        return job.result()

    def get_raw_from_counts(self, counts):
        raw = []
        for key in counts:
            for i in range(counts[key]):
                raw.append(key)
        return raw

    def get_counts_from_raw(self):
        counts = {}
        for bitstring in self.raw_data:
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def _get_z_exp(self):
        Z = []
        Zi = []

        counts = self.get_counts_from_raw()
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
        # print(self.zi)
        return sum(Z)

    def get_all_z_exp(self, counts=None):
        # This function generates all possible Pauli strings of length `num_qubits` using 'Z' and 'I'
        pauli_strings = []

        # Loop over the range 0 to 2^num_qubits - 1
        for i in range(1, 2 ** self.n):
            # Convert number to binary, remove '0b' prefix, and zfill to make it num_qubits long
            binary_str = bin(i)[2:].zfill(self.n)

            # Replace '0' with 'I' and '1' with 'Z'
            pauli_str = binary_str.replace('0', 'I').replace('1', 'Z')

            pauli_strings.append(pauli_str)

        values = []
        for pauli_string in pauli_strings:
            values.append(self.get_zn_exp(pauli_string, counts=counts))
        return values

    def get_z_nearest_neighbors(self, counts=None):
        # This function generates Pauli strings for single qubits and pairs of nearest neighbors
        pauli_strings = []

        # Single qubit measurements
        for i in range(self.n):
            pauli_str = 'I' * i + 'Z' + 'I' * (self.n - i - 1)
            pauli_strings.append(pauli_str)

        # Nearest neighbor pairs
        for i in range(self.n - 1):
            pauli_str = 'I' * i + 'ZZ' + 'I' * (self.n - i - 2)
            pauli_strings.append(pauli_str)

        values = []
        for pauli_string in pauli_strings:
            values.append(self.get_zn_exp(pauli_string, counts=counts))
        return values

    def get_n_nearest_neighbors(self, n_neighbors, counts=None):
        # This function generates Pauli strings for single qubits and n nearest neighbors
        pauli_strings = []

        # Single qubit measurements
        for i in range(self.n):
            pauli_str = 'I' * i + 'Z' + 'I' * (self.n - i - 1)
            pauli_strings.append(pauli_str)

        # n nearest neighbor pairs
        for distance in range(1, n_neighbors + 1):
            for i in range(self.n - distance):
                pauli_str = 'I' * i + 'Z' + 'I' * (distance - 1) + 'Z' + 'I' * (self.n - i - distance - 1)
                pauli_strings.append(pauli_str)

        values = []
        for pauli_string in pauli_strings:
            values.append(self.get_zn_exp(pauli_string, counts=counts))
        return values

    def get_zn_exp(self, pauli_string, counts=None):
        """
        Calculate the expectation value of an observable represented by a Pauli string,
        correctly handling the product of observables.

        :param counts: dictionary of counts
        :param pauli_string: A string representing the observable, composed of 'Z' and 'I'.
        :return: The expectation value of the observable.
        """
        if counts is None:
            counts = self.get_counts_from_raw()
        normalization = sum(counts.values())
        expectation_value = 0

        for outcome, count in counts.items():
            # outcome = outcome[::-1]  # Reversing to match qubit order
            term_value = 1

            for i, pauli_char in enumerate(pauli_string):
                if pauli_char == 'Z':
                    # Apply Z operation
                    term_value *= 1 if outcome[i] == '0' else -1
                elif pauli_char != 'I':
                    # Invalid character in Pauli string
                    raise ValueError(f"Invalid character '{pauli_char}' in Pauli string. Only 'Z' and 'I' are allowed.")

            expectation_value += term_value * count

        return expectation_value / normalization

    def add_decay_raw(self):
        new_data = []
        decay = self.L
        # decay.reverse()
        # print(np.exp(-decay[2] * self.delay))

        for bitstring in self.raw_data:
            new_bitstring = ""
            for i, bit in enumerate(bitstring):
                p = 0.5 * (1 - np.exp(-decay[i] * self.delay))
                if bit == '0':
                    new_bitstring += '0' if np.random.random() > p else '1'
                else:
                    new_bitstring += '1' if np.random.random() > p else '0'
            new_data.append(new_bitstring)
        self.raw_data = new_data
        self._get_z_exp()

    def add_noise_raw(self):
        self.add_noise()
        new_data = []
        for key in self.noise_model:
            bitstring = key
            for i in range(self.noise_model[key]):
                new_data.append(bitstring)
        random.shuffle(new_data)
        self.raw_data = self.raw_data + new_data
        self._get_z_exp()

    def to_int(self):
        new_data = []
        for bitstring in self.raw_data:
            new_data.append(int(bitstring, 2))
        self.raw_data = new_data


class RamseyBatch:

    def __init__(self, RamseyExperiments: "list of RamseyExperiment"):
        self.dist = None
        self.J = None
        self.W = None
        self.L = None
        self.J_fit = []
        self.decay_fit = []
        self.W_fit = []
        self.delay = []
        self.Z = []
        self.Zi = []
        self.n = None
        self.fft_data = []
        self.frequencies = None
        self.zi_formated = []
        self.qubits_measured = None

        self.RamseyExperiments = RamseyExperiments
        for RamseyExperiment in RamseyExperiments:
            if self.J is None:
                self.W = RamseyExperiment.W
                self.L = RamseyExperiment.L
                self.J = RamseyExperiment.J
                self.n = RamseyExperiment.n
                self.qubits_measured = RamseyExperiment.qubits_measured
            self.delay.append(RamseyExperiment.delay)
            self.Z.append(RamseyExperiment.z)
            self.Zi.append(RamseyExperiment.zi)
        for i in range(self.n):
            self.zi_formated.append([sublist[i] for sublist in self.Zi])

    def get_zi(self, n):
        return self.zi_formated[n]

    def apply_lowpass_filter(self, data, cutoff=3):
        def design_lowpass_filter(cutoff, fs, numtaps):
            nyquist = 0.5 * fs
            normalized_cutoff = cutoff / nyquist
            return signal.firwin(numtaps, normalized_cutoff)

        filter_coefficients = design_lowpass_filter(cutoff, len(self.delay) / self.delay[-1], len(self.delay) * 2)
        return np.convolve(data, filter_coefficients, mode='same')

    def fft(self, data=None, filter=None):
        def single_fft(data):
            if isinstance(data, np.ndarray):
                extended = np.concatenate([data[::-1], data])
            else:
                extended = data[::-1] + data
            if filter is not None:
                extended = self.apply_lowpass_filter(extended)
            fft_output = np.fft.fft(extended)
            sample_rate = len(self.delay) / self.delay[-1]
            frequencies = np.fft.fftfreq(len(extended), 1 / sample_rate)
            frequencies *= (2 * np.pi)

            paired = sorted(zip(frequencies, fft_output))
            frequencies, fft_output = zip(*paired)
            if self.frequencies is None:
                self.frequencies = np.array(frequencies)
            return np.abs(fft_output)

        if data is not None:
            return single_fft(data)

        for i in range(self.n):
            self.fft_data.append(single_fft(self.get_zi(i)))
        return self.frequencies, self.fft_data

    # def apply_filter(self, filter,params):
    #     zi = []
    #     for i in range(self.n):
    #         zi.append(filter(self.get_zi(i)))
    #     self.zi_formated = zi

    def set_zi(self, zi):
        self.zi_formated = zi

    def fit_to_theory(self, data=None):
        def model_func(x, a, w):
            return (np.cos(w * x)) * np.exp(-a * x)

        if data is None:
            data = []
            for i in range(self.n):
                data.append(self.get_zi(i))

        parameters = []
        for i in range(len(data)):
            initial_guess = [1, 1]
            # Perform the curve fitting
            x_points = self.delay
            y_points = data[i]
            try:
                params, params_covariance, *c = curve_fit(model_func, x_points, y_points, p0=initial_guess)
            except:
                params = [100, 100]
            # self.decay_fit.append(params[0])
            # self.W_fit.append(params[1])
            parameters.append(np.abs(params))
        return parameters

    def log_likelihood_estimator(self):
        params = []

        def f(t, w, a, x):
            if x == 1:
                val = 0.5 * (1 + np.cos(w * t) * np.exp(-a * t))
            if x == -1:
                val = 0.5 * (1 - np.cos(w * t) * np.exp(-a * t))

            if val == 0:
                return 1e-10
            return val

        for i in range(self.n):
            def log_likelihood(params):
                w, a = params[0], params[1]
                likelihood = 0
                for j in range(len(self.delay)):
                    shots = self.RamseyExperiments[j].shots
                    ups = shots * (self.get_zi(i)[j] + 1) / 2
                    downs = shots - ups
                    likelihood += ups * np.log(f(self.delay[j], w, a, 1))
                    likelihood += downs * np.log(f(self.delay[j], w, a, -1))
                return -likelihood

            initial_params = np.array([5, 5])
            result = minimize(log_likelihood, initial_params)
            if result.success:
                w0 = result.x[0]
                gamma = result.x[1]
                params.append({'gamma': gamma, 'w0': w0})

            else:
                params.append({'gamma': np.nan, 'w0': np.nan})
        return params

    def calc_dist(self, fitted_values, correct_values):
        fitted_values = np.array(fitted_values)
        correct_values = np.array(correct_values)
        mse = (fitted_values - correct_values) ** 2 / len(fitted_values)
        precent_error = (np.sqrt(np.abs(mse)) / np.abs(correct_values)) * 100
        return precent_error


def complex_fit(batch_x, batch_y):
    def model_func(t, a, w):
        x_model = (np.cos(w * t)) * np.exp(-a * t)
        y_model = -(np.sin(w * t)) * np.exp(-a * t)
        return np.concatenate([x_model, y_model])

    data_x = []
    data_y = []

    for i in range(batch_x.n):
        data_x.append(batch_x.get_zi(i))
        data_y.append(batch_y.get_zi(i))

    parameters = []
    for i in range(len(data_x)):
        initial_guess = [1, 1]
        # Perform the curve fitting
        t_points = batch_x.delay
        z_points = np.concatenate([np.array(data_x[i]), np.array(data_y[i])])
        try:
            params, params_covariance, *c = curve_fit(model_func, t_points, z_points, p0=initial_guess)
        except:
            params = [100, 100]
        # self.decay_fit.append(params[0])
        # self.W_fit.append(params[1])
        parameters.append(params)
    return parameters


def full_complex_fit(batch_x, batch_y):
    def model_func(times, *params):
        n = batch_x.n
        A = params[:n]
        W = params[n:2 * n]
        J = params[2 * n:2 * n + n - 1]
        x_models = []
        y_models = []

        x_correlations = []
        y_correlations = []

        X = []
        Y = []
        for t in times:
            for i in range(n):
                x_model = (np.cos(W[i] * t)) * np.exp(-A[i] * t)
                y_model = -(np.sin(W[i] * t)) * np.exp(-A[i] * t)
                norm = 1
                if i > 0:
                    x_model += np.cos((W[i] + J[i - 1]) * t) * np.exp(-A[i] * t)
                    y_model -= np.sin((W[i] + J[i - 1]) * t) * np.exp(-A[i] * t)
                    norm += 1

                if i < n - 1:
                    x_model += np.cos((W[i] + J[i]) * t) * np.exp(-A[i] * t)  # Updated to use J[i] instead of J[i + 1]
                    y_model -= np.sin((W[i] + J[i]) * t) * np.exp(-A[i] * t)  # Same here, for consistency
                    norm += 1

                if 0 < i < n - 1:
                    x_model += np.cos((W[i] + J[i - 1] + J[i]) * t) * np.exp(-A[i] * t)
                    y_model -= np.sin((W[i] + J[i - 1] + J[i]) * t) * np.exp(-A[i] * t)
                    norm += 1
                x_models.append(x_model / norm)
                y_models.append(y_model / norm)

                X.append(x_model / norm)
                Y.append(y_model / norm)
            for i in range(n - 1):
                x_correlation = 0
                norm = 0
                if i < len(J):
                    x_correlation += (np.cos((W[i] + W[i + 1] + J[i]) * t)) * np.exp(- (A[i] + A[i + 1]) * t)
                    x_correlation += (np.cos((W[i] - W[i + 1]) * t)) * np.exp(-2 * A[i] * t)
                    norm += 2
                    if i + 1 < len(J):
                        x_correlation += (np.cos((W[i] + W[i + 1] + J[i] + J[i + 1]) * t)) * np.exp(
                            -(A[i] + A[i + 1]) * t)
                        x_correlation += (np.cos((W[i + 1] - W[i] + J[i + 1]) * t)) * np.exp(-(A[i] + A[i + 1]) * t)
                        norm += 2

                if i - 1 >= 0:
                    x_correlation += (np.cos((W[i] - W[i + 1] + J[i - 1]) * t)) * np.exp(-(A[i] + A[i + 1]) * t)
                    x_correlation += (np.cos((W[i] + W[i + 1] + J[i] + J[i - 1]) * t)) * np.exp(-(A[i] + A[i + 1]) * t)
                    norm += 2
                    if i + 1 < len(J):
                        x_correlation += (np.cos((W[i] - W[i + 1] + J[i - 1] - J[i + 1]) * t)) * np.exp(
                            -(A[i] + A[i + 1]) * t)
                        x_correlation += (np.cos((W[i] + W[i + 1] + J[i - 1] + J[i] + J[i + 1]) * t)) * np.exp(
                            -(A[i] + A[i + 1]) * t)
                        norm += 2

                y_correlation = 0
                norm = 0
                if i < len(J):
                    y_correlation -= (np.cos((W[i] + W[i + 1] + J[i]) * t)) * np.exp(-(A[i] + A[i + 1]) * t)
                    y_correlation += (np.cos((W[i] - W[i + 1]) * t)) * np.exp(-(A[i] + A[i + 1]) * t)
                    norm += 2
                    if i + 1 < len(J):
                        y_correlation += (np.cos((W[i + 1] - W[i] + J[i + 1]) * t)) * np.exp(-(A[i] + A[i + 1]) * t)
                        y_correlation -= (np.cos((W[i] + W[i + 1] + J[i] + J[i + 1]) * t)) * np.exp(
                            -(A[i] + A[i + 1]) * t)
                        norm += 2

                if i - 1 >= 0:
                    y_correlation += (np.cos((W[i] - W[i + 1] + J[i - 1]) * t)) * np.exp(-(A[i] + A[i + 1]) * t)
                    y_correlation -= (np.cos((W[i] + W[i + 1] + J[i] + J[i - 1]) * t)) * np.exp(-(A[i] + A[i + 1]) * t)
                    norm += 2
                    if i + 1 < len(J):
                        y_correlation += (np.cos((W[i] - W[i + 1] + J[i - 1] - J[i + 1]) * t)) * np.exp(
                            -(A[i] + A[i + 1]) * t)
                        y_correlation -= (np.cos((W[i] + W[i + 1] + J[i - 1] + J[i] + J[i + 1]) * t)) * np.exp(
                            -(A[i] + A[i + 1]) * t)
                        norm += 2

                y_correlations.append(y_correlation / norm)
                x_correlations.append(x_correlation / norm)

                X.append(x_correlation / norm)
                Y.append(y_correlation / norm)

            # return np.concatenate([x_models, x_correlations,y_models, y_correlations])
        return np.concatenate([X, Y])

    data_x = []
    data_y = []
    # cor_x = []
    # cor_y = []

    for i in range(batch_x.n):
        # data_x.append(batch_x.get_zi(i))
        # data_y.append(batch_y.get_zi(i))
        data_x = [exp.get_n_nearest_neighbors(1) for exp in batch_x.RamseyExperiments]
        data_y = [exp.get_n_nearest_neighbors(1) for exp in batch_y.RamseyExperiments]

    # initial_guess = [1] * (2 * batch_x.n + (batch_x.n - 1))  # Adjusted to include J
    initial_guess = [random.random() for i in range(2 * batch_x.n + (batch_x.n - 1))]
    bounds_lower = [-2 * np.pi] * (2 * batch_x.n + (batch_x.n - 1))
    bounds_upper = [2 * np.pi] * (2 * batch_x.n + (batch_x.n - 1))
    bounds = (bounds_lower, bounds_upper)

    # Perform the curve fitting
    t_points = batch_x.delay
    x_points = np.concatenate(data_x)
    y_points = np.concatenate(data_y)
    z_points = np.concatenate([x_points, y_points])
    params, params_covariance, *c = curve_fit(model_func, t_points, z_points, p0=initial_guess, bounds=bounds,
                                              method='trf')
    guessed_decay = params[:batch_x.n][::-1]
    guessed_W = params[batch_x.n:2 * batch_x.n][::-1]
    guessed_J = params[2 * batch_x.n:3 * batch_x.n - 1][::-1]
    return guessed_decay, guessed_W, guessed_J

