import csv
from datetime import datetime
import multiprocessing
import os
import random
import sys
import threading

from tqdm import tqdm
import pandas as pd
import numpy as np

import Ramsey_ExperimentV2
import Ramsey_ExperimentV3
import estimator

# import ramsey_experiment

# total_experiments = 1
total_time = 0.5 * np.pi


# total_time_stamps = 2
# number_of_qubits = 2
# shots = 500
# # time_stamps = [0.0922, 0.35693]
# time_stamps = np.linspace(0, total_time, total_time_stamps)
# file_name = 'experiments.csv'


def run_experiment(position, qubits, total_experiments, total_time_stamps, shots, mean_decay, mean_w, mean_j, std,
                   correlations, Gamma_1, Gamma_2,
                   filename='experiments.csv'):
    experiments_x = []
    experiments_y = []
    scale = 0.05
    time_stamps = np.linspace(0, total_time, total_time_stamps + 1)
    time_stamps = np.delete(time_stamps, 0)
    time_stamps = [0.5]

    def create_csv_from_experiments(experiments, decay, W, J, filename, Gamma_1, Gamma_2):
        # Open the file in write mode
        filename = filename + ".csv"
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            # Create a CSV writer
            csv_writer = csv.writer(file)
            Gamma_1_list = [Gamma_1 for _ in range(qubits)]
            Gamma_2_list = [Gamma_2 for _ in range(qubits)]

            # Prepare the header
            header = []
            experiments_x = experiments[0]
            experiments_y = experiments[1]
            max_length = max(len(experiment) for experiment in experiments_x)
            for i in range(len(time_stamps)):
                for j in range(len(experiments_x[0])):
                    header.append(f'X_{i}_{j}')
            for i in range(len(time_stamps)):
                for j in range(len(experiments_x[0])):
                    header.append(f'Y_{i}_{j}')
            for i in range(len(decay[0])):
                header.append(f'decay_{i}')
            for i in range(len(W[0])):
                header.append(f'W_{i}')
            for i in range(len(J[0])):
                header.append(f'J_{i}')
            for i in range(len(Gamma_1_list)):
                header.append(f'Gamma_1_{i}')
            for i in range(len(Gamma_2_list)):
                header.append(f'Gamma_2_{i}')

            csv_writer.writerow(header)

            # Write each experiment's data
            for j in range(int(len(experiments_x) / 2)):
                row = []
                # for part in experiments_x:
                row.extend(experiments_x[j])
                row.extend(experiments_y[j])
                for k in decay[j]:
                    row.append(k)
                for k in W[j]:
                    row.append(k)
                for k in J[j]:
                    row.append(k)
                for k in Gamma_1_list:
                    row.append(k)
                for k in Gamma_2_list:
                    row.append(k)
                csv_writer.writerow(row)

    W_parameters = []
    J_parameters = []
    decay_parameters = []
    with tqdm(total=total_experiments, file=sys.stdout, dynamic_ncols=True, position=position,
              desc=f'Experiments for {filename}') as pbar:
        for i in range(total_experiments):
            experiment_parts = []
            L = [random.gauss(mean_decay, mean_decay / 3) for _ in range(qubits)]
            W = [random.gauss(mean_w, std) for _ in range(qubits)]
            J = [random.gauss(mean_j, std) for _ in range(qubits - 1)]
            time = [0.8]

            W_parameters.append(W)
            J_parameters.append(J)
            decay_parameters.append(L)
            # batch_x, batch_y = Ramsey_ExperimentV2.ramsey_global(qubits, shots, time_stamps, L, W, J)
            # exp_x = [batch_x.RamseyExperiments[i] for i in range(len(time_stamps))]
            # exp_y = [batch_y.RamseyExperiments[i] for i in range(len(time_stamps))]
            J_dict = {}
            for i in range(qubits - 1):  # Only connect to the next neighbor
                J_dict[(i, i + 1)] = J[i]

            Gamma_1_list = [Gamma_1 for _ in range(qubits)]
            Gamma_2_list = [Gamma_2 for _ in range(qubits)]

            batch_x_det, batch_y_det, batch_x_cross, batch_y_cross = Ramsey_ExperimentV3.ramsey_local(qubits, shots,
                                                                                                      time, W, J_dict,
                                                                                                      Gamma_1_list,
                                                                                                      Gamma_2_list, L)

            # values_x_det = [exp.get_n_nearest_neighbors(correlations) for exp in batch_x_det.RamseyExperiments][0]
            # values_y_det = [exp.get_n_nearest_neighbors(correlations) for exp in batch_y_det.RamseyExperiments][0]
            # values_x_cross = [exp.get_n_nearest_neighbors(correlations) for exp in batch_x_cross.RamseyExperiments][0]
            # values_y_cross = [exp.get_n_nearest_neighbors(correlations) for exp in batch_y_cross.RamseyExperiments][0]

            values_x_det = np.concatenate(np.transpose(batch_x_det.zi).tolist())
            values_y_det = np.concatenate(np.transpose(batch_y_det.zi).tolist())
            values_x_cross = np.concatenate(np.transpose(batch_x_cross.zi).tolist())
            values_y_cross = np.concatenate(np.transpose(batch_y_cross.zi).tolist())


            values_x = []
            values_y = []

            values_x.extend(values_x_det)
            values_x.extend(values_x_cross)
            values_y.extend(values_y_det)
            values_y.extend(values_y_cross)

            experiments_x.append(values_x)
            experiments_y.append(values_y)

            pbar.update(1)
    create_csv_from_experiments((experiments_x, experiments_y), decay_parameters, W_parameters, J_parameters,
                                filename, Gamma_1, Gamma_2)


# run_experiment(number_of_qubits, time_stamps, shots, total_experiments, filename=file_name)


def read_excel_to_variables(file_path):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    # Create a dictionary to hold the variables
    variables = {}

    # Iterate through each column and assign it to a variable
    for column in df.columns:
        variables[column] = df[column].values

    return variables


# File path for the uploaded Excel file
file_path = 'Data_generator_template.xlsx'

# Read and store the data from the Excel file
variables = read_excel_to_variables(file_path)


def process_experiment(args):
    run_experiment(*args)


def main():
    # Prepare the arguments for each experiment
    current_datetime_with_minutes = datetime.now().strftime('%Y-%m-%d_%H-%M')
    directory_name_current_datetime_with_minutes = f"Data/{current_datetime_with_minutes}"
    os.makedirs(directory_name_current_datetime_with_minutes, exist_ok=False)

    experiment_args = [
        (
            variables["Qubits"][i],
            variables["Total_Experiments"][i],
            variables["Time_Stamps"][i],
            variables["Shots"][i],
            variables["Mean_Decay"][i],
            variables["Mean_W"][i],
            variables["Mean_J"][i],
            variables["Std"][i],
            variables["Gamma_1"][i],
            variables["Gamma_2"][i],
            variables["Correlations"][i],
            directory_name_current_datetime_with_minutes + "/" + variables["File_Name"][i]
        )
        for i in range(len(variables["Qubits"]))
    ]

    # Create a list to hold the processes
    processes = []
    # Create and start a process for each set of argumentsf
    for i, args in enumerate(experiment_args):
        args = (i,) + args
        process = multiprocessing.Process(target=process_experiment, args=(args,))
        processes.append(process)
        process.start()
        # pool.apply_async(process_experiment, args=(args,))

    # Wait for all processes to complete
    for process in processes:
        process.join()

    print("Done!")


if __name__ == '__main__':
    main()
