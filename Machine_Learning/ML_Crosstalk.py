import pandas as pd
import os
import tensorflow as tf


def load_training_data(location) -> pd.DataFrame:
    # Get a list of all CSV files in the directory, excluding test.csv
    csv_files = [f for f in os.listdir(location) if f.endswith('.csv') and f != 'test.csv']

    # Initialize an empty list to hold DataFrames
    df_list = []

    # Loop through the list of CSV files
    for csv_file in csv_files:
        # Create a DataFrame from the CSV file and append it to the list
        df = pd.read_csv(os.path.join(location, csv_file))
        df_list.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df


def split_data(df, test_size=0.1) -> (pd.DataFrame, pd.DataFrame):
    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    # Calculate the number of rows that will be used for testing
    test_rows = int(test_size * df.shape[0])

    # Split the DataFrame into training and testing sets
    test_df = df.iloc[:test_rows]
    train_df = df.iloc[test_rows:]

    return train_df, test_df


def load_test_data(location) -> pd.DataFrame:
    # Load the test data from the test.csv file
    test_df = pd.read_csv(os.path.join(location, 'test.csv'))

    return test_df


def get_output_keys(n) -> list:
    decay = ['decay_{}'.format(i) for i in range(n)]
    W = ['W_{}'.format(i) for i in range(n)]
    J = ['J_{}'.format(i) for i in range(n - 1)]
    return decay + W + J


def build_model(input_layer, num_layers, output_keys, nodes_per_layer=None, default_nodes=64) -> tf.keras.layers.Layer:
    # If no list of nodes is provided, use the default number of nodes for all layers
    if nodes_per_layer is None:
        nodes_per_layer = [default_nodes] * num_layers

    # Check if nodes_per_layer has the correct number of layers
    assert len(nodes_per_layer) == num_layers, "Length of nodes_per_layer must match num_layers"

    # Build the hidden layers dynamically
    hidden_layer = input_layer
    for i in range(num_layers):
        hidden_layer = tf.keras.layers.Dense(nodes_per_layer[i], activation='relu')(hidden_layer)

    # Output layer (Assuming len(output_keys) is predefined)
    output = tf.keras.layers.Dense(len(output_keys))(hidden_layer)

    return output


def get_keys(df, qubits):
    all_keys = df.keys().tolist()
    output_keys = get_output_keys(qubits)  # ['decay_0', 'decay_1', 'W_0', 'W_1', "J_0"] ...
    input_keys = [key for key in all_keys if key not in output_keys]
    return input_keys, output_keys
###get output keys
