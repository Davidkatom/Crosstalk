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


def ramsey_local(n, total_shots, delay, W, J, Gamma_1, Gamma_2, Gamma_phi):
    Gamma_phi = np.array(Gamma_phi) / 2  # TODO this is for testing (gamma_phi = 2 decay rate)
    state_det_0_string = ""
    state_det_1_string = ""
    state_cross_0_string = ["0"] * (2 * n)
    state_cross_1_string = ["0"] * (2 * n)

    measurements_det_0 = []
    measurements_det_1 = []
    measurements_cross_0 = []
    measurements_cross_1 = []

    total_shots = int(total_shots / len(delay))
    total_shots = int(total_shots / 8)

    # Create initial states
    for i in range(n):
        if i % 2 == 0:
            state_det_0_string += "+"
            measurements_det_0.append(i)
            state_det_1_string += "0"
        else:
            state_det_0_string += "0"
            state_det_1_string += "+"
            measurements_det_1.append(i)

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

    for i in range(n):
        if state_cross_0_string[i] == "+":
            measurements_cross_0.append(i)
        if state_cross_1_string[i] == "+":
            measurements_cross_1.append(i)

    state_cross_0_string = "".join(state_cross_0_string)
    state_cross_1_string = "".join(state_cross_1_string)

    # Evolve the states
    H = ramsey_H(n, W, J)
    c_o = c_ops(Gamma_1, Gamma_2, Gamma_phi, n)

    state_det_0 = create_state(state_det_0_string)
    state_det_1 = create_state(state_det_1_string)
    state_cross_0 = create_state(state_cross_0_string)
    state_cross_1 = create_state(state_cross_1_string)

    modif_delay = False
    if delay[0] != 0:
        delay = np.insert(delay, 0, 0.0)
        modif_delay = True

    # evolved_det0 = mesolve(H, state_det_0, delay, c_o, [])


    try:
        evolved_det0 = mesolve(H, state_det_0, delay, c_o, [])
    except IntegratorException as e:
        print("IntegratorException occurred!")
        print("Error details:", e)
        print("Hamiltonian (H):", H)
        print("Initial State (state_det_0):", state_det_0)
        print("Time Delay (delay):", delay)
        print("Collapse Operators (c_o):", c_o)
        print("W = ", W)
        print("J = ", J)
        print("L = ", Gamma_phi)

    evolved_det1 = mesolve(H, state_det_1, delay, c_o, [])
    evolved_cross0 = mesolve(H, state_cross_0, delay, c_o, [])
    evolved_cross1 = mesolve(H, state_cross_1, delay, c_o, [])

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

    batch_x_det, batch_y_det, batch_x_cross, batch_y_cross = Ramsey_batch(), Ramsey_batch(), Ramsey_batch(), Ramsey_batch()
    batch_x_det.n = n
    batch_y_det.n = n
    batch_x_cross.n = n
    batch_y_cross.n = n

    batch_x_det.total_shots = total_shots
    batch_y_det.total_shots = total_shots
    batch_x_cross.total_shots = total_shots
    batch_y_cross.total_shots = total_shots

    batch_x_det.delay = delay
    batch_y_det.delay = delay
    batch_x_cross.delay = delay
    batch_y_cross.delay = delay

    batch_x_det.W = W
    batch_y_det.W = W
    batch_x_cross.W = W
    batch_y_cross.W = W

    J_list = J.items()

    batch_x_det.J = J_list
    batch_y_det.J = J_list
    batch_x_cross.J = J_list
    batch_y_cross.J = J_list

    batch_x_det.Gamma_1 = Gamma_1
    batch_y_det.Gamma_1 = Gamma_1
    batch_x_cross.Gamma_1 = Gamma_1
    batch_y_cross.Gamma_1 = Gamma_1

    batch_x_det.Gamma_2 = Gamma_2
    batch_y_det.Gamma_2 = Gamma_2
    batch_x_cross.Gamma_2 = Gamma_2
    batch_y_cross.Gamma_2 = Gamma_2

    batch_x_det.Gamma_phi = Gamma_phi
    batch_y_det.Gamma_phi = Gamma_phi
    batch_x_cross.Gamma_phi = Gamma_phi
    batch_y_cross.Gamma_phi = Gamma_phi

    batch_x_det.zi = expectation_det_x
    batch_y_det.zi = expectation_det_y
    batch_x_cross.zi = expectation_cross_x
    batch_y_cross.zi = expectation_cross_y

    batch_x_det.qubits_measured = measured_qubits
    batch_y_det.qubits_measured = measured_qubits
    batch_x_cross.qubits_measured = measured_qubits
    batch_y_cross.qubits_measured = measured_qubits

    return batch_x_det, batch_y_det, batch_x_cross, batch_y_cross
