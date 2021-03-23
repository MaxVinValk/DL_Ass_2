import tensorflow as tf
from tensorflow.keras import layers

'''
    Sampling layer
        Uses the inputs as parameters to sample from normal distributions
'''


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit"""

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


'''
  2D Replication Padding
  Attributes:
    - padding: (padding_width, padding_height) tuple
'''


class ReplicationPadding2D(layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReplicationPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]], 'SYMMETRIC')
