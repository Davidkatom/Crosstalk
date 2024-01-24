# custom_layer.py
import tensorflow as tf
import numpy as np

class CustomConnectedLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_groups, activation=None, **kwargs):
        """
        Custom layer with restricted connectivity.
        :param units: Total number of neurons in the layer.
        :param num_groups: Number of groups to divide the neurons into.
        :param activation: Activation function to use.
        """
        super(CustomConnectedLayer, self).__init__(**kwargs)
        self.units = units
        self.num_groups = num_groups
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        """
        Create the weight matrix with custom connectivity.
        """
        # Calculate neurons per group
        neurons_per_group = self.units // self.num_groups

        # Initialize the weight matrix with zeros
        init_shape = (input_shape[-1], self.units)
        initializer = tf.zeros_initializer()
        self.kernel = self.add_weight(
            shape=init_shape,
            initializer=initializer,
            trainable=True,
            name='kernel'
        )

        # Create custom connections
        for i in range(self.num_groups):
            start_row = i * neurons_per_group
            end_row = (i + 1) * neurons_per_group
            self.kernel[start_row:end_row, i * neurons_per_group:(i + 1) * neurons_per_group].assign(
                tf.keras.initializers.GlorotUniform()(shape=(neurons_per_group, neurons_per_group))
            )

        # Bias
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def call(self, inputs):
        """
        Forward pass through the layer.
        """
        output = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation is not None:
            return self.activation(output)
        return output


class CustomConnectedLayer2(tf.keras.layers.Layer):
    def __init__(self, units, n, k, activation=None, **kwargs):
        super(CustomConnectedLayer2, self).__init__(**kwargs)
        self.units = units
        self.n = n
        self.k = k
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]), self.units],
                                      initializer='uniform',
                                      trainable=True)
        # Custom connections
        self.mask = np.zeros([int(input_shape[-1]), self.units])
        for i in range(0, int(input_shape[-1]), self.n):
            for j in range(0, self.units, self.k):
                self.mask[i:i + self.n, j:j + self.k] = 1

        self.bias = self.add_weight("bias",
                                    shape=[self.units, ],
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs):
        masked_kernel = self.kernel * self.mask
        return tf.matmul(inputs, masked_kernel) + self.bias



# Dummy input for visualization
dummy_input = tf.keras.Input(shape=(10,))
dummy_layer = CustomConnectedLayer(5, 2, activation='relu')
dummy_output = dummy_layer(dummy_input)

# Creating a model for visualization
model = tf.keras.Model(inputs=dummy_input, outputs=dummy_output)

# Visualize the model
model.summary()