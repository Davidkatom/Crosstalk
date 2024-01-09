import csv
import random
from tqdm import tqdm

import numpy as np
import Ramsey_ExperimentV2

total_experiments = 100000
total_time = 0.5 * np.pi
measurement_amount = 10
number_of_qubits = 2
shots = 100
time_stamps = np.linspace(0, total_time, measurement_amount)

experiments = []


def create_csv_from_experiments(experiments, decay, W, J, filename='experiments.csv'):
    # Open the file in write mode
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        # Create a CSV writer
        csv_writer = csv.writer(file)

        # Prepare the header
        header = []
        max_length = max(len(experiment) for experiment in experiments)
        for i in range(max_length):
            for j in range(len(experiments[0][i])):
                header.append(f'{i}_{j}')
        for i in range(len(decay[0])):
            header.append(f'decay_{i}')
        for i in range(len(W[0])):
            header.append(f'W_{i}')
        for i in range(len(J[0])):
            header.append(f'J_{i}')

        # Write the header
        csv_writer.writerow(header)

        # Write each experiment's data
        for j, experiment in enumerate(experiments):
            row = []
            for part in experiment:
                row.extend(part)
            for k in decay[j]:
                row.append(k)
            for k in W[j]:
                row.append(k)
            for k in J[j]:
                row.append(k)
            csv_writer.writerow(row)


W_parameters = []
J_parameters = []
decay_parameters = []
for i in tqdm(range(total_experiments), desc='total experiments'):
    experiment_parts = []
    L = [random.gauss(1, 1) for _ in range(number_of_qubits)]
    W = [random.gauss(3, 1) for _ in range(number_of_qubits)]
    J = [random.gauss(3, 1) for _ in range(number_of_qubits - 1)]
    W_parameters.append(W)
    J_parameters.append(J)
    decay_parameters.append(L)
    for t in time_stamps:
        exp = Ramsey_ExperimentV2.RamseyExperiment(number_of_qubits, t, shots, J, W, L)
        exp.create_full_circuit()
        # exp.add_decay_raw()
        # exp.add_noise_raw()
        exp.to_int()
        experiment_parts.append(exp.raw_data)
    experiments.append(experiment_parts)
create_csv_from_experiments(experiments, decay_parameters, W_parameters, J_parameters, filename='experiments.csv')
