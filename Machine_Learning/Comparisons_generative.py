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
location = "C:\Projects\Crosstalk\Machine_Learning\Data/final/2qubits-many experiments-TODO/"



def compare(location):
    # train_df = ML_Crosstalk.load_training_data(location)
    # test_df = ML_Crosstalk.load_test_data(location)
    data = pd.read_csv(location)
    train_df, test_df = ML_Crosstalk.split_data(data, 0.1)
    learning_rate = 0.0001
    epochs = 1000
    batch_size = 500

    output_keys, input_keys = ML_Crosstalk.get_keys(train_df, qubits)  # Reversed

    inputs = {key: tf.keras.layers.Input(shape=(1,), name=key) for key in input_keys}
    concatenated_inputs = tf.keras.layers.concatenate(list(inputs.values()))

    # Prepare data for training
    train_features = {key: train_df[key] for key in inputs}
    train_labels = train_df[output_keys]

    # Similarly prepare test and validation data
    test_features = {key: test_df[key] for key in inputs}
    test_labels = test_df[output_keys]

    nodes_per_layer = [32, 64, 64, 64, 64, 64, 64, 8]  # Optional, can be None
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

    new_data = test_df


    # Initialize the input parameters as variables to optimize
    initial_guess = {
        'decay_0': 0.5,
        'decay_1': 0.5,
        'W_0': 0,
        'W_1': 0,
        'J_0': 0
    }
    input_vars = {key: tf.Variable(value, dtype=tf.float32) for key, value in initial_guess.items()}

    # Use the variables as inputs to the model
    def model_loss(target):
        # Pass the variables through the model to get the predicted output
        model_inputs = {key: tf.expand_dims(input_vars[key], 0) for key in input_keys}
        predicted_outputs = model(model_inputs, training=False)

        # Calculate the loss between predicted and target outputs
        loss = tf.reduce_mean(tf.abs(predicted_outputs - target))
        return loss * 1000

    # Set up the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    errors = []
    for i in range(10):
        line = new_data.iloc[i]
        correct_output = line[output_keys].array
        parameters = {key: line[key] for key in input_keys}
        input_vars = {key: tf.Variable(value, dtype=tf.float32) for key, value in initial_guess.items()}
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

        def model_loss(target):
            # Pass the variables through the model to get the predicted output
            model_inputs = {key: tf.expand_dims(input_vars[key], 0) for key in input_keys}
            predicted_outputs = model(model_inputs, training=False)

            # Calculate the loss between predicted and target outputs
            loss = tf.reduce_mean(tf.abs(predicted_outputs - target))
            return loss

        for step in range(500):  # Adjust the number of steps as needed
            with tf.GradientTape() as tape:
                loss = model_loss(correct_output)

            # Compute the gradients of the loss with respect to the input variables
            grads = tape.gradient(loss, input_vars.values())

            # Apply the gradients to the input variables
            optimizer.apply_gradients(zip(grads, input_vars.values()))

            # Print the loss and current input variables every 100 steps
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.numpy()}, Inputs: {[input_vars[key].numpy() for key in input_keys]}")
                print(np.array(list(parameters.values())))

        optimized_inputs = {key: var.numpy() for key, var in input_vars.items()}

        optimized_inputs = np.array(list(optimized_inputs.values()))
        parameters = np.array(list(parameters.values()))

        # optimized_inputs = optimized_inputs[:-1]
        # parameters = parameters[:-1]

        error = estimator.percent_error(optimized_inputs, parameters)
        errors.append(error)
    print("The mean percent error is: ", np.mean(error) * 100)
    return np.mean(error) * 100


#for each file in the directory
errors = []
for filename in os.listdir(location):
    if filename.endswith(".csv"):
        # print(filename)
        #compare the file
        error = compare(location + filename)
        errors.append((filename, error))

for error in errors:
    print(error)
    print("\n")