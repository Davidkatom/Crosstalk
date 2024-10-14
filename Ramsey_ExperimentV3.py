import random
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip_qip.operations import hadamard_transform, snot, phasegate


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
        H += 0.5 * W[i] * (Z_i - identity)

    for (i, j), J_ij in J.items():
        Z_i = tensor([sigmaz() if n == i else qeye(2) for n in range(N)])
        Z_j = tensor([sigmaz() if n == j else qeye(2) for n in range(N)])
        H += (1 / 8) * J_ij * (Z_i - identity) * (Z_j - identity)
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


def ramsey_local(n, total_shots, delay, W, J, Gamma_1, Gamma_2, Gamma_phi):
    state_det_0 = ""
    state_det_1 = ""
    state_cross_0 = ""
    state_cross_1 = ""

    measurements_det_0 = []
    measurements_det_1 = []
    measurements_cross_0 = []
    measurements_cross_1 = []

    # Create initial states
    for i in range(n):
        if i % 2 == 0:
            state_det_0 += "+"
            measurements_det_0.append(i)
            state_det_1 += "0"
        else:
            state_det_0 += "0"
            state_det_1 += "+"
            measurements_det_1.append(i)
    for i in range(n):
        if (i + 1) % 4 == 0:
            state_cross_0 += "1"
        elif i % 2 == 0:
            state_cross_0 += "+"
            measurements_cross_0.append(i)
        else:
            state_cross_0 += "0"
    for i in range(n):
        if (i + 3) % 4 == 0:
            state_cross_1 += "+"
            measurements_cross_1.append(i)
        elif i % 2 == 0:
            state_cross_1 += "1"
        else:
            state_cross_1 += "0"
    # Evolve the states
    H = ramsey_H(n, W, J)
    c_o = c_ops(Gamma_1, Gamma_2, Gamma_phi, n)

    state_det_0 = create_state(state_det_0)
    state_det_1 = create_state(state_det_1)
    state_cross_0 = create_state(state_cross_0)
    state_cross_1 = create_state(state_cross_1)

    evolved_det0 = mesolve(H, state_det_0, delay, c_o, [])
    evolved_det1 = mesolve(H, state_det_1, delay, c_o, [])
    evolved_cross0 = mesolve(H, state_cross_0, delay, c_o, [])
    evolved_cross1 = mesolve(H, state_cross_1, delay, c_o, [])

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
    for i in range(n):
        expectation_det_x1 = calculate_expectation(measurements_det_x_0[i], "".join(
            ["X" if i in measurements_det_0 else "I" for i in range(n)]))
        expectation_det_x2 = calculate_expectation(measurements_det_x_1[i], "".join(
            ["X" if i in measurements_det_1 else "I" for i in range(n)]))
        joint = [item for pair in zip(expectation_det_x1, expectation_det_x2) for item in pair]
        expectation_det_x.append(joint)

    for i in range(n):
        expectation_det_y1 = calculate_expectation(measurements_det_y_0[i], "".join(
            ["Y" if i in measurements_det_0 else "I" for i in range(n)]))
        expectation_det_y2 = calculate_expectation(measurements_det_y_1[i], "".join(
            ["Y" if i in measurements_det_1 else "I" for i in range(n)]))
        joint = [item for pair in zip(expectation_det_y1, expectation_det_y2) for item in pair]
        expectation_det_y.append(joint)


    for i in range(0, n, 2):
        index = 4 * int(i / 4) + int((i / 2) % 2)
        ##### TODO continue here

# "X" if i in measurements_det_0 else "I" for i in range(n)]
