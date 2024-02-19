import math

import numpy as np
import qiskit.quantum_info as qi


def sample_from_statevector(vector, shots):
    vector = np.square(np.abs(vector))
    counts = {}
    for i in range(shots):
        choice = np.random.choice(range(len(vector)), p=vector)
        bitstring = format(choice, 'b').zfill(len(vector))
        if bitstring in counts:
            counts[bitstring] += 1
        else:
            counts[bitstring] = 1
    return counts


def sample_from_circuit(circuit, shots):
    op = qi.Operator(circuit)
    N = int(math.log2(op.dim[0]))  # op.dim[0] gives the number of rows
    zero_state = np.zeros(2 ** N)
    zero_state[0] = 1
    state_vector = op.data @ zero_state
    return sample_from_statevector(state_vector, shots)

