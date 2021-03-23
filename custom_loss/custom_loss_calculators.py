from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import keras

from custom_loss.fpl import FPL

class LossCalculator(ABC):
    @abstractmethod
    def calculate_loss(self, **kwargs):
        pass


class KLDCalculator(LossCalculator):

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def calculate_loss(self, **kwargs):
        kl_loss = -0.5 * (1 + kwargs["z_log_var"] - tf.square(kwargs["z_mean"]) - tf.exp(kwargs["z_log_var"]))
        return tf.multiply(tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)), self.alpha)


class FPLCalculator(LossCalculator):

    def __init__(self, input_shape, batch_size, loss_layers, beta):
        self.fpl = FPL(input_shape=input_shape, batch_size=batch_size,
                       loss_layers=loss_layers, beta=beta
        )

    def calculate_loss(self, **kwargs):
        return self.fpl.calculate_fp_loss(kwargs["original"], kwargs["reconstruction"])


class ReconLossCalculator(LossCalculator):
    def calculate_loss(self, **kwargs):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(
                kwargs["original"], kwargs["reconstruction"]), axis=(1, 2))
        )

        return reconstruction_loss