import random
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip_qip.operations import hadamard_transform, snot, phasegate
from qutip.solver.integrator import IntegratorException


class Ramsey_batch:
    def __init__(self):
        self.n = 0
        self.total_shots = 0
        self.delay = []
        self.W = []
        self.J = {}
        self.L = []
        self.Gamma_1 = []
        self.Gamma_2 = []
        self.zi = []
        self.qubits_measured = []

    def get_zi(self, i):
        return [x[i] for x in self.zi]


# Function to create the initial state based on a string
def create_state(state_string):
    state_list = []
    for c in state_string:
        if c == "0":
            state_list.append(basis(2, 0))
        elif c == "1":
            state_list.append(basis(2, 1))
        elif c == "+":
            plus_state = (basis(2, 0) + basis(2, 1)).unit()
            state_list.append(plus_state)
        else:
            raise ValueError(f"Invalid character '{c}' in state string. Allowed characters are '0', '1', '+'.")
    return tensor(state_list)


def ramsey_H(N, W, J):
    # Construct the Hamiltonian H0
    H = 0
    identity = tensor([qeye(2) for _ in range(N)])
    for i in range(N):
        Z_i = tensor([sigmaz() if n == i else qeye(2) for n in range(N)])
        X_i = tensor([sigmax() if n == i else qeye(2) for n in range(N)])
        H -= 0.5 * W[i] * (Z_i - identity)

    for (i, j), J_ij in J.items():
        Z_i = tensor([sigmaz() if n == i else qeye(2) for n in range(N)])
        Z_j = tensor([sigmaz() if n == j else qeye(2) for n in range(N)])

        X_i = tensor([sigmax() if n == i else qeye(2) for n in range(N)])
        X_j = tensor([sigmax() if n == j else qeye(2) for n in range(N)])
        H += (1 / 4) * J_ij * (Z_i - identity) * (Z_j - identity)
        # H += (1 / 4) * J_ij * (X_i - identity) * (X_j - identity)
    return H


def c_ops(Gamma_1, Gamma_2, Gamma_phi, N):
    # Define collapse operators
    c_ops = []
    for i in range(N):
        # sigma_+ operator on qubit i
        sp_i = tensor([sigmap() if n == i else qeye(2) for n in range(N)])
        # sigma_- operator on qubit i
        sm_i = tensor([sigmam() if n == i else qeye(2) for n in range(N)])
        # sigma_z operator on qubit i
        sz_i = tensor([sigmaz() if n == i else qeye(2) for n in range(N)])

        if Gamma_1[i] != 0:
            c_ops.append(np.sqrt(Gamma_1[i]) * sp_i)
        if Gamma_2[i] != 0:
            c_ops.append(np.sqrt(Gamma_2[i]) * sm_i)
        if Gamma_phi[i] != 0:
            c_ops.append(np.sqrt(Gamma_phi[i]) * sz_i)
    return c_ops


def evolve_state(state: str, H, c_ops, t: list[float]):
    psi = create_state(state)
    return mesolve(H, psi, t, c_ops, [])


def get_expectation(result):
    N = len(result.states[0])
    expectations_X = []
    expectation_Y = []
    for i in range(N):
        X_i = tensor([sigmax() if n == i else qeye(2) for n in range(N)])
        Y_i = tensor([sigmay() if n == i else qeye(2) for n in range(N)])
        exp_X_i = expect(X_i, result.states)
        exp_Y_i = expect(Y_i, result.states)
        expectations_X.append(exp_X_i)
        expectation_Y.append(exp_Y_i)
    return expectations_X, expectation_Y


def sample_measurements(rho, num_shots, measurement_basis):
    """
    Simulates measurements on the density matrix rho in the specified measurement basis and returns counts of outcomes.

    Parameters:
    rho (Qobj): Density matrix of the system.
    num_shots (int): Number of measurement shots.
    measurement_basis (str): String specifying the measurement basis for each qubit (e.g., 'XXZI').

    Returns:
    counts (dict): Dictionary with measurement outcomes as keys and counts as values.
    """
    N = int(np.log2(rho.shape[0]))  # Number of qubits
    if len(measurement_basis) != N:
        raise ValueError("Measurement basis string length must match the number of qubits.")

    # Apply rotation operators to rho based on the measurement basis
    rotation_ops = []
    measured_qubits = []
    for i, basis_char in enumerate(measurement_basis):
        if basis_char == 'X':
            # Rotate to X-basis using Hadamard gate
            rotation_ops.append(hadamard_transform(1))
            measured_qubits.append(i)
        elif basis_char == 'Y':
            # Rotate to Y-basis using S† H (where S† is the adjoint of the phase gate)
            S_dag = phasegate(-np.pi / 2)
            H = snot()
            rotation_ops.append(H * S_dag)
            measured_qubits.append(i)
        elif basis_char == 'Z':
            # No rotation needed for Z-basis measurement
            rotation_ops.append(qeye(2))
            measured_qubits.append(i)
        elif basis_char == 'I':
            # Identity operator; qubit is not measured
            rotation_ops.append(qeye(2))
        else:
            raise ValueError(
                f"Invalid character '{basis_char}' in measurement basis. Allowed characters are 'X', 'Y', 'Z', 'I'.")

    # Build the total rotation operator
    U = tensor(rotation_ops)
    # Rotate the density matrix to the measurement basis
    rho_rotated = U * rho * U.dag()
    rho_rotated = rho_rotated.ptrace(measured_qubits)
    rho_rotated = Qobj(rho_rotated.full().reshape(2 ** len(measured_qubits), 2 ** len(measured_qubits)),
                       dims=[[2 ** len(measured_qubits)], [2 ** len(measured_qubits)]])

    # Trace out qubits that are not measured
    if len(measured_qubits) == 0:
        raise ValueError("At least one qubit must be measured.")

    # Generate the list of computational basis states for the measured qubits
    num_measured = len(measured_qubits)
    basis_states = []
    for i in range(2 ** num_measured):
        state = basis(2 ** num_measured, i)
        basis_states.append(state)

    # Calculate the probabilities for each basis state
    probs = []
    for state in basis_states:
        prob = (state.dag() * rho_rotated * state).real
        probs.append(prob)

    # Normalize probabilities in case of numerical inaccuracies
    probs = np.array(probs)
    probs /= probs.sum()

    # Generate measurement outcomes
    outcomes = np.random.choice(len(basis_states), size=num_shots, p=probs)

    # Convert outcomes to bit strings
    outcome_strings = [format(i, '0{}b'.format(num_measured)) for i in outcomes]

    # Map measured bits back to the full qubit system
    full_outcome_strings = []
    for bits in outcome_strings:
        full_bits = list('-' * N)
        for idx, qubit_idx in enumerate(measured_qubits):
            full_bits[qubit_idx] = bits[idx]
        full_outcome_strings.append(''.join(full_bits))

    # Count the occurrences
    counts = {}
    for outcome in full_outcome_strings:
        counts[outcome] = counts.get(outcome, 0) + 1

    return counts


def calculate_expectation(counts, pauli_string):
    """
    Calculates the expectation value of the Pauli operator specified by pauli_string based on measurement counts.

    Parameters:
    counts (dict): Dictionary with measurement outcomes as keys and counts as values.
    pauli_string (str): String specifying the Pauli operator for each qubit (e.g., 'XXZI').

    Returns:
    expectation_value (float): The expectation value of the specified Pauli operator.
    """
    N = len(pauli_string)
    total_shots = sum(counts.values())
    expectation = 0.0

    # Map measurement outcomes to eigenvalues
    for outcome, count in counts.items():
        eigenvalue = 1
        for i, pauli in enumerate(pauli_string):
            bit = int(outcome[i])
            if pauli in ('X', 'Y', 'Z'):
                if bit == 0:
                    eigenval = 1
                else:
                    eigenval = -1
                eigenvalue *= eigenval
            elif pauli == 'I':
                eigenvalue *= 1  # Identity operator has eigenvalue 1
        expectation += eigenvalue * count

    expectation_value = expectation / total_shots
    return expectation_value


def sample_state(states, shots: int, measurement: str):
    Counts = []
    for state in states:
        # print(result.states[i].data.todense())
        counts = sample_measurements(state, shots, measurement)
        Counts.append(counts)
    return Counts


def ramsey_local(n, total_shots, delay, Gamma_phi, W, J, Gamma_1=None, Gamma_2=None):
    Gamma_phi = np.array(Gamma_phi) / 2  # TODO this is for testing (gamma_phi = 2 decay rate)
    total_shots = total_shots / len(delay)
    if n != 1:
        total_shots = int(total_shots / 8)
    else:
        total_shots = int(total_shots / 2)


    state_det_0_string, state_det_1_string = create_detuning_states(n)
    state_cross_0_string, state_cross_1_string = create_crosstalk_states(n)


    # Evolve the states
    H = ramsey_H(n, W, J)
    if Gamma_1 is None:
        Gamma_1 = [0] * n
        Gamma_2 = [0] * n
    c_o = c_ops(Gamma_1, Gamma_2, Gamma_phi, n)

    modif_delay = False
    if delay[0] != 0:
        delay = np.insert(delay, 0, 0.0)
        modif_delay = True

    evolved_det0 = mesolve(H, create_state(state_det_0_string), delay, c_o, [])
    evolved_det1 = mesolve(H, create_state(state_det_1_string), delay, c_o, [])
    evolved_cross0 = mesolve(H, create_state(state_cross_0_string), delay, c_o, [])
    evolved_cross1 = mesolve(H, create_state(state_cross_1_string), delay, c_o, [])

    if modif_delay:
        delay = delay[1:]
        evolved_det0.states = evolved_det0.states[1:]
        evolved_det1.states = evolved_det1.states[1:]
        evolved_cross0.states = evolved_cross0.states[1:]
        evolved_cross1.states = evolved_cross1.states[1:]

    # Sample the states
    measurements_det_x_0 = sample_state(evolved_det0.states, total_shots, "X" * n)
    measurements_det_x_1 = sample_state(evolved_det1.states, total_shots, "X" * n)
    measurements_det_y_0 = sample_state(evolved_det0.states, total_shots, "Y" * n)
    measurements_det_y_1 = sample_state(evolved_det1.states, total_shots, "Y" * n)
    measurements_cross_x_0 = sample_state(evolved_cross0.states, total_shots, "X" * n)
    measurements_cross_x_1 = sample_state(evolved_cross1.states, total_shots, "X" * n)
    measurements_cross_y_0 = sample_state(evolved_cross0.states, total_shots, "Y" * n)
    measurements_cross_y_1 = sample_state(evolved_cross1.states, total_shots, "Y" * n)

    # Calculate the expectation values
    expectation_det_x = []
    expectation_det_y = []
    expectation_cross_x = []
    expectation_cross_y = []
    # Detuning
    for i in range(len(delay)):
        snapshot_x = []
        snapshot_y = []
        for j in range(n):
            pauli_X = j * "I" + "X" + (n - j - 1) * "I"
            pauli_Y = j * "I" + "Y" + (n - j - 1) * "I"
            if j % 2 == 0:
                snapshot_x.append(calculate_expectation(measurements_det_x_0[i], pauli_X))
                snapshot_y.append(calculate_expectation(measurements_det_y_0[i], pauli_Y))
            else:
                snapshot_x.append(calculate_expectation(measurements_det_x_1[i], pauli_X))
                snapshot_y.append(calculate_expectation(measurements_det_y_1[i], pauli_Y))
        expectation_det_x.append(snapshot_x)
        expectation_det_y.append(snapshot_y)

    # Crosstalk
    measured_qubits = [0] * n
    for i in range(len(delay)):
        snapshot_x = [0] * n
        snapshot_y = [0] * n
        for j in range(0, n):
            pauli_X = j * "I" + "X" + (n - j - 1) * "I"
            pauli_Y = j * "I" + "Y" + (n - j - 1) * "I"

            if state_cross_0_string[j] == "+":
                if state_cross_0_string[j + 1] == "1":
                    index = j
                else:
                    index = j - 1

                snapshot_x[index] = calculate_expectation(measurements_cross_x_0[i], pauli_X)
                snapshot_y[index] = calculate_expectation(measurements_cross_y_0[i], pauli_Y)
                measured_qubits[index] = j
            if state_cross_1_string[j] == "+":
                if state_cross_1_string[j + 1] == "1":
                    index = j
                else:
                    index = j - 1

                snapshot_x[index] = calculate_expectation(measurements_cross_x_1[i], pauli_X)
                snapshot_y[index] = calculate_expectation(measurements_cross_y_1[i], pauli_Y)
                measured_qubits[index] = j
        expectation_cross_x.append(snapshot_x)
        expectation_cross_y.append(snapshot_y)

    expectations = [expectation_det_x, expectation_det_y, expectation_cross_x, expectation_cross_y]
    return package_data(expectations, n, total_shots, delay, W, J, Gamma_1, Gamma_2, Gamma_phi, measured_qubits)


def ramsey_local_gausian(n, total_shots, optimal_time,sigma, delay, Gamma_phi, W, J, Gamma_1=None, Gamma_2=None):
    Gamma_phi = np.array(Gamma_phi) / 2  # TODO this is for testing (gamma_phi = 2 decay rate)
    if n != 1:
        total_shots = int(total_shots / 8)
    else:
        total_shots = int(total_shots / 2)
    # Compute the probability density at each delay time
    probabilities = np.exp(-0.5 * ((delay - optimal_time) / sigma) ** 2)

    # Normalize the probabilities so that sum(probabilities) = 1
    probabilities /= probabilities.sum()

    # Compute the number of shots for each delay time
    shots_distribution = probabilities * total_shots

    # Since we need integer numbers of shots, we can take the floor
    integer_shots = np.floor(shots_distribution).astype(int)

    # Compute the remaining shots to allocate
    remaining_shots = total_shots - integer_shots.sum()

    # If there are remaining shots, allocate them to the bins with the highest fractional parts
    fractional_parts = shots_distribution - integer_shots
    sorted_indices = np.argsort(fractional_parts)  # Indices of fractional parts in descending order

    for i in range(int(remaining_shots)):
        idx = sorted_indices[i % len(sorted_indices)]
        integer_shots[idx] += 1

    # Now, integer_shots sum up to total_shots
    shots_distribution = integer_shots




    state_det_0_string, state_det_1_string = create_detuning_states(n)
    state_cross_0_string, state_cross_1_string = create_crosstalk_states(n)


    # Evolve the states
    H = ramsey_H(n, W, J)
    if Gamma_1 is None:
        Gamma_1 = [0] * n
        Gamma_2 = [0] * n
    c_o = c_ops(Gamma_1, Gamma_2, Gamma_phi, n)

    modif_delay = False
    if delay[0] != 0:
        delay = np.insert(delay, 0, 0.0)
        modif_delay = True

    evolved_det0 = mesolve(H, create_state(state_det_0_string), delay, c_o, [])
    evolved_det1 = mesolve(H, create_state(state_det_1_string), delay, c_o, [])
    evolved_cross0 = mesolve(H, create_state(state_cross_0_string), delay, c_o, [])
    evolved_cross1 = mesolve(H, create_state(state_cross_1_string), delay, c_o, [])

    if modif_delay:
        delay = delay[1:]
        evolved_det0.states = evolved_det0.states[1:]
        evolved_det1.states = evolved_det1.states[1:]
        evolved_cross0.states = evolved_cross0.states[1:]
        evolved_cross1.states = evolved_cross1.states[1:]

    # Sample the states

    measurements_det_x_0 = []
    measurements_det_x_1 = []
    measurements_det_y_0 = []
    measurements_det_y_1 = []
    measurements_cross_x_0 = []
    measurements_cross_x_1 = []
    measurements_cross_y_0 = []
    measurements_cross_y_1 = []

    new_delay = []
    for i in range(len(delay)):
        if shots_distribution[i] == 0:
            continue
        new_delay.append(delay[i])
        measurements_det_x_0.append(sample_state([evolved_det0.states[i]], shots_distribution[i], "X" * n))
        measurements_det_x_1.append(sample_state([evolved_det1.states[i]], shots_distribution[i], "X" * n))
        measurements_det_y_0.append(sample_state([evolved_det0.states[i]], shots_distribution[i], "Y" * n))
        measurements_det_y_1.append(sample_state([evolved_det1.states[i]], shots_distribution[i], "Y" * n))
        measurements_cross_x_0.append(sample_state([evolved_cross0.states[i]], shots_distribution[i], "X" * n))
        measurements_cross_x_1.append(sample_state([evolved_cross1.states[i]], shots_distribution[i], "X" * n))
        measurements_cross_y_0.append(sample_state([evolved_cross0.states[i]], shots_distribution[i], "Y" * n))
        measurements_cross_y_1.append(sample_state([evolved_cross1.states[i]], shots_distribution[i], "Y" * n))
    delay = new_delay
    measurements_det_x_0 = np.concatenate(measurements_det_x_0)
    measurements_det_x_1 = np.concatenate(measurements_det_x_1)
    measurements_det_y_0 = np.concatenate(measurements_det_y_0)
    measurements_det_y_1 = np.concatenate(measurements_det_y_1)
    measurements_cross_x_0 = np.concatenate(measurements_cross_x_0)
    measurements_cross_x_1 = np.concatenate(measurements_cross_x_1)
    measurements_cross_y_0 = np.concatenate(measurements_cross_y_0)
    measurements_cross_y_1 = np.concatenate(measurements_cross_y_1)


    # Calculate the expectation values
    expectation_det_x = []
    expectation_det_y = []
    expectation_cross_x = []
    expectation_cross_y = []
    # Detuning
    for i in range(len(delay)):
        snapshot_x = []
        snapshot_y = []
        for j in range(n):
            pauli_X = j * "I" + "X" + (n - j - 1) * "I"
            pauli_Y = j * "I" + "Y" + (n - j - 1) * "I"
            if j % 2 == 0:
                snapshot_x.append(calculate_expectation(measurements_det_x_0[i], pauli_X))
                snapshot_y.append(calculate_expectation(measurements_det_y_0[i], pauli_Y))
            else:
                snapshot_x.append(calculate_expectation(measurements_det_x_1[i], pauli_X))
                snapshot_y.append(calculate_expectation(measurements_det_y_1[i], pauli_Y))
        expectation_det_x.append(snapshot_x)
        expectation_det_y.append(snapshot_y)

    # Crosstalk
    measured_qubits = [0] * n
    for i in range(len(delay)):
        snapshot_x = [0] * n
        snapshot_y = [0] * n
        for j in range(0, n):
            pauli_X = j * "I" + "X" + (n - j - 1) * "I"
            pauli_Y = j * "I" + "Y" + (n - j - 1) * "I"

            if state_cross_0_string[j] == "+":
                if state_cross_0_string[j + 1] == "1":
                    index = j
                else:
                    index = j - 1

                snapshot_x[index] = calculate_expectation(measurements_cross_x_0[i], pauli_X)
                snapshot_y[index] = calculate_expectation(measurements_cross_y_0[i], pauli_Y)
                measured_qubits[index] = j
            if state_cross_1_string[j] == "+":
                if state_cross_1_string[j + 1] == "1":
                    index = j
                else:
                    index = j - 1

                snapshot_x[index] = calculate_expectation(measurements_cross_x_1[i], pauli_X)
                snapshot_y[index] = calculate_expectation(measurements_cross_y_1[i], pauli_Y)
                measured_qubits[index] = j
        expectation_cross_x.append(snapshot_x)
        expectation_cross_y.append(snapshot_y)

    expectations = [expectation_det_x, expectation_det_y, expectation_cross_x, expectation_cross_y]
    return package_data(expectations, n, total_shots, delay, W, J, Gamma_1, Gamma_2, Gamma_phi, measured_qubits)


def ramsey_local_X(n, total_shots, delay, Gamma_phi, W, J, Gamma_1=None, Gamma_2=None):
    Gamma_phi = np.array(Gamma_phi) / 2  # TODO this is for testing (gamma_phi = 2 decay rate)
    total_shots = total_shots / len(delay)
    if n != 1:
        total_shots = total_shots / 4
    total_shots = int(total_shots)
    state_det_0_string, state_det_1_string = create_detuning_states(n)
    state_cross_0_string, state_cross_1_string = create_crosstalk_states(n)


    # Evolve the states
    H = ramsey_H(n, W, J)
    if Gamma_1 is None:
        Gamma_1 = [0] * n
        Gamma_2 = [0] * n
    c_o = c_ops(Gamma_1, Gamma_2, Gamma_phi, n)

    modif_delay = False
    if delay[0] != 0:
        delay = np.insert(delay, 0, 0.0)
        modif_delay = True

    evolved_det0 = mesolve(H, create_state(state_det_0_string), delay, c_o, [])
    evolved_det1 = mesolve(H, create_state(state_det_1_string), delay, c_o, [])
    evolved_cross0 = mesolve(H, create_state(state_cross_0_string), delay, c_o, [])
    evolved_cross1 = mesolve(H, create_state(state_cross_1_string), delay, c_o, [])

    if modif_delay:
        delay = delay[1:]
        evolved_det0.states = evolved_det0.states[1:]
        evolved_det1.states = evolved_det1.states[1:]
        evolved_cross0.states = evolved_cross0.states[1:]
        evolved_cross1.states = evolved_cross1.states[1:]

    # Sample the states
    measurements_det_x_0 = sample_state(evolved_det0.states, total_shots, "X" * n)
    measurements_det_x_1 = sample_state(evolved_det1.states, total_shots, "X" * n)
    measurements_cross_x_0 = sample_state(evolved_cross0.states, total_shots, "X" * n)
    measurements_cross_x_1 = sample_state(evolved_cross1.states, total_shots, "X" * n)

    # Calculate the expectation values
    expectation_det_x = []
    expectation_cross_x = []
    # Detuning
    for i in range(len(delay)):
        snapshot_x = []
        for j in range(n):
            pauli_X = j * "I" + "X" + (n - j - 1) * "I"
            if j % 2 == 0:
                snapshot_x.append(calculate_expectation(measurements_det_x_0[i], pauli_X))
            else:
                snapshot_x.append(calculate_expectation(measurements_det_x_1[i], pauli_X))
        expectation_det_x.append(snapshot_x)

    # Crosstalk
    measured_qubits = [0] * n
    for i in range(len(delay)):
        snapshot_x = [0] * n
        for j in range(0, n):
            pauli_X = j * "I" + "X" + (n - j - 1) * "I"

            if state_cross_0_string[j] == "+":
                if state_cross_0_string[j + 1] == "1":
                    index = j
                else:
                    index = j - 1

                snapshot_x[index] = calculate_expectation(measurements_cross_x_0[i], pauli_X)
                measured_qubits[index] = j
            if state_cross_1_string[j] == "+":
                if state_cross_1_string[j + 1] == "1":
                    index = j
                else:
                    index = j - 1

                snapshot_x[index] = calculate_expectation(measurements_cross_x_1[i], pauli_X)
                measured_qubits[index] = j
        expectation_cross_x.append(snapshot_x)

    expectations = [expectation_det_x, expectation_cross_x]
    return package_data(expectations, n, total_shots, delay, W, J, Gamma_1, Gamma_2, Gamma_phi, measured_qubits)


def ramsey_global(n, total_shots, delay, Gamma_phi, W, J, Gamma_1=None, Gamma_2=None):
    Gamma_phi = np.array(Gamma_phi) / 2  # TODO this is for testing (gamma_phi = 2 decay rate)
    total_shots = int(total_shots / len(delay))
    total_shots = int(total_shots / 2)

    state = "+" * n

    H = ramsey_H(n, W, J)
    if Gamma_1 is None:
        Gamma_1 = [0] * n
        Gamma_2 = [0] * n
    c_o = c_ops(Gamma_1, Gamma_2, Gamma_phi, n)

    modif_delay = False
    if delay[0] != 0:
        delay = np.insert(delay, 0, 0.0)
        modif_delay = True

    evolved_state = mesolve(H, create_state(state), delay, c_o, [])
    if modif_delay:
        delay = delay[1:]
        evolved_state.states = evolved_state.states[1:]

    # Sample the states
    measurements_x = sample_state(evolved_state.states, total_shots, "X" * n)
    measurements_y = sample_state(evolved_state.states, total_shots, "Y" * n)

    # Calculate the expectation values
    expectation_x = []
    expectation_y = []
    # Detuning
    for i in range(len(delay)):
        snapshot_x = []
        snapshot_y = []
        for j in range(n):
            pauli_X = j * "I" + "X" + (n - j - 1) * "I"
            pauli_Y = j * "I" + "Y" + (n - j - 1) * "I"
            snapshot_x.append(calculate_expectation(measurements_x[i], pauli_X))
            snapshot_y.append(calculate_expectation(measurements_y[i], pauli_Y))

            expectation_x.append(snapshot_x)
            expectation_y.append(snapshot_y)
    measured_qubits = [0] * n
    expectations = [expectation_x, expectation_y]
    return package_data(expectations, n, total_shots, delay, W, J, Gamma_1, Gamma_2, Gamma_phi, measured_qubits)


def package_data(expectations, n, total_shots, delay, W, J, Gamma_1, Gamma_2, Gamma_phi, measured_qubits):
    batches = []
    for exp in expectations:
        batch = Ramsey_batch()
        batch.n = n
        batch.total_shots = total_shots
        batch.delay = delay
        batch.W = W
        batch.J = J.items()
        batch.Gamma_1 = Gamma_1
        batch.Gamma_2 = Gamma_2
        batch.Gamma_phi = Gamma_phi
        batch.zi = exp
        batch.qubits_measured = measured_qubits
        batches.append(batch)
    return tuple(batches)


def create_detuning_states(n):
    state_det_0_string = ""
    state_det_1_string = ""
    for i in range(n):
        if i % 2 == 0:
            state_det_0_string += "+"
            # measurements_det_0.append(i)
            state_det_1_string += "0"
        else:
            state_det_0_string += "0"
            state_det_1_string += "+"
            # measurements_det_1.append(i)

    return state_det_0_string, state_det_1_string


def create_crosstalk_states(n):
    state_cross_0_string = ["0"] * (2 * n)
    state_cross_1_string = ["0"] * (2 * n)
    for i in range(n):
        if (i - 1) % 4 == 0:
            state_cross_0_string[i - 1] = "+"
            state_cross_0_string[i] = "1"
            state_cross_0_string[i + 1] = "+"
        if (i - 3) % 4 == 0:
            state_cross_1_string[i - 1] = "+"
            state_cross_1_string[i] = "1"
            state_cross_1_string[i + 1] = "+"
    state_cross_0_string = state_cross_0_string[:n]
    state_cross_1_string = state_cross_1_string[:n]

    for i in range(n - 1):
        if state_cross_0_string[i + 1] == "+" and state_cross_0_string[i] == "+":
            state_cross_0_string[i + 1] = "0"
        if state_cross_1_string[i + 1] == "+" and state_cross_1_string[i] == "+":
            state_cross_1_string[i + 1] = "0"

    state_cross_0_string = "".join(state_cross_0_string)
    state_cross_1_string = "".join(state_cross_1_string)

    return state_cross_0_string, state_cross_1_string


def run_exp():
    initial_state = ''
