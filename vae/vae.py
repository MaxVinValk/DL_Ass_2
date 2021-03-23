import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from cnn.fpl import FPL
from vae.custom_layers import Sampling, ReplicationPadding2D


class VAE(keras.Model):
    def __init__(self, input_shape, latent_dim, fpl: FPL, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder, pre_flatten_shape = self.__create_encoder(input_shape, latent_dim)
        self.decoder = self.__create_decoder(latent_dim, pre_flatten_shape)
        self.fpl = fpl
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.fp_loss_tracker = keras.metrics.Mean(name="fp_loss")

    def __create_encoder(self, input_shape, latent_dim):
        encoder_inputs = keras.Input(shape=input_shape)

        x = encoder_inputs

        x = layers.Conv2D(32, (4, 4), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(64, (4, 4), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(128, (4, 4), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(256, (4, 4), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        pre_flatten_shape = tf.keras.backend.int_shape(x)[1:]

        x = layers.Flatten()(x)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

        z = Sampling()([z_mean, z_log_var])

        enc = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        return enc, pre_flatten_shape

    def __create_decoder(self, latent_dim, pre_flatten_shape):
        latent_inputs = keras.Input(shape=(latent_dim))

        x = layers.Dense(np.prod(pre_flatten_shape))(latent_inputs)
        x = layers.Reshape(pre_flatten_shape)(x)

        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = layers.Conv2D(128, (3, 3))(x)
        x = ReplicationPadding2D(padding=(1, 1))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = layers.Conv2D(64, (3, 3))(x)
        x = ReplicationPadding2D(padding=(1, 1))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = layers.Conv2D(32, (3, 3))(x)
        x = ReplicationPadding2D(padding=(1, 1))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = layers.Conv2D(128, (3, 3))(x)
        x = ReplicationPadding2D(padding=(1, 1))(x)
        x = layers.BatchNormalization()(x)
        decoder_outputs = layers.Conv2D(3, (1, 1))(x)

        dec = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        return dec

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.fp_loss_tracker,
            self.kl_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)

            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(
                    data, reconstruction), axis=(1, 2))
            )
            fp_loss = self.fpl.calculate_fp_loss(data, reconstruction)

            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = tf.add(kl_loss, fp_loss)
            # total_loss = reconstruction_loss + kl_loss
            # print('\n\nFP Loss')
            # print(fp_loss, fp_loss.numpy())
            # print('\n\nKL LOSS')
            # print(kl_loss, kl_loss.numpy())

            # print(total_loss, total_loss.numpy())
            # exit()

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            self.fp_loss_tracker.update_state(fp_loss)

            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "fp_loss": self.fp_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()
            }

    def train(self, data, epochs, log_dir):

        # The optimizer calls the schedule once per train_step = 1 batch,
        # we only want to change the learning rate after a batch, and we want the
        # learning rate to remain the same in between these adjustments
        learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0005,
            decay_steps=data.get_batches_per_epoch(),
            decay_rate=0.5,
            staircase=True
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)

        self.compile(optimizer=keras.optimizers.Adam(
            learning_rate=0.0005))

        self.fit(data.get_dataset(), epochs=epochs, batch_size=data.get_batch_size(), callbacks=[tensorboard_callback])

    def save(self, folder):
        self.encoder.save(f"{folder}/enc/")
        self.decoder.save(f"{folder}/dec/")

