import os

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import estimator

import ML_Crosstalk

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.6f}".format
qubits = 2
# date = "2024-09-05_18-52"
# name = "Qubits_2_Lines_10000_TS_2_Shots_10000_MeanDecay_0.75_MeanW_0_MeanJ_0_Std_1_Correlations_0_7.csv"
location = "C:\Projects\Crosstalk\Machine_Learning\Data/final/largeT1/"


def compare(location):
    # train_df = ML_Crosstalk.load_training_data(location)
    # test_df = ML_Crosstalk.load_test_data(location)
    data = pd.read_csv(location)
    train_df, test_df = ML_Crosstalk.split_data(data, 0.05)

    train_df.head()
    print(len(train_df))

    from tensorflow.python.keras.regularizers import L2, L1

    learning_rate = 0.0001
    epochs = 700
    batch_size = 1000

    # Get all column names as a list
    input_keys, output_keys = ML_Crosstalk.get_keys(train_df, qubits)

    inputs = {key: tf.keras.layers.Input(shape=(1,), name=key) for key in input_keys}
    concatenated_inputs = tf.keras.layers.concatenate(list(inputs.values()))

    # Prepare data for training
    train_features = {key: train_df[key] for key in inputs}
    train_labels = train_df[output_keys]

    # Similarly prepare test and validation data
    test_features = {key: test_df[key] for key in inputs}
    test_labels = test_df[output_keys]

    nodes_per_layer = [32, 64, 64, 64, 64, 64, 64, 64, 8]  # Optional, can be None
    output = ML_Crosstalk.build_model(concatenated_inputs, len(nodes_per_layer), output_keys, nodes_per_layer)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_absolute_error')
    model.summary()

    # Train the model
    # history = model.fit(train_features, train_labels, validation_data=(validation_features, validation_labels), epochs=epochs, batch_size=batch_size);
    history = model.fit(train_features, train_labels, validation_split=0.2, epochs=epochs, batch_size=batch_size)

    # Evaluate the model
    model.evaluate(test_features, test_labels)

    # Extract loss and validation loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    new_data = test_df

    error = []
    sample_size = 100
    for i in range(sample_size):
        line = new_data.iloc[i]

        # correct_output = line[['decay_0', 'decay_1', 'W_0', 'W_1', "J_0"]].array
        correct_output = line[
            ML_Crosstalk.get_output_keys(qubits)].array

        input_data = {key: np.array([line[key]]) for key in inputs}
        predictions = model.predict(input_data)
        error.append(estimator.percent_error(predictions[0], correct_output))

    print("The mean percent error is: ", np.median(error) * 100)
    print("The std percent error is: ", np.std(error) * 100)
    return np.median(error) * 100, np.std(error) * 100


# for each file in the directory
errors = []
stds = []
for filename in os.listdir(location):
    if filename.endswith(".csv"):
        # print(filename)
        # compare the file
        error, std = compare(location + filename)
        errors.append((filename, error))
        errors.append(std)

for error in errors:
    print(error)
    print("\n")
