from abc import ABC

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from vae.custom_layers import Sampling, ReplicationPadding2D


class VAEArchitecture(ABC):

    def __init__(self):
        self.encoder = None
        self.decoder = None

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def save(self, folder):
        try:
            self.encoder.save(f"{folder}/enc/")
            self.decoder.save(f"{folder}/dec/")
        except Exception as e:
            print("Failed to save VAE architecture")
            print(e)


class PaperArchitecture(VAEArchitecture):

    def __init__(self, input_shape, latent_dim):
        self.encoder, pre_flatten_shape = self.__create_encoder(
            input_shape, latent_dim)
        self.decoder = self.__create_decoder(latent_dim, pre_flatten_shape)

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

        enc = keras.Model(
            encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        return enc, pre_flatten_shape

    def __create_decoder(self, latent_dim, pre_flatten_shape):
        latent_inputs = keras.Input(shape=(latent_dim))

        x = layers.Dense(np.prod(pre_flatten_shape))(latent_inputs)
        x = layers.Reshape(pre_flatten_shape)(x)

        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = ReplicationPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(128, (3, 3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = ReplicationPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(64, (3, 3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = ReplicationPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(32, (3, 3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = ReplicationPadding2D(padding=(1, 1))(x)
        decoder_outputs = layers.Conv2D(3, (3, 3))(x)

        dec = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        return dec


class TutorialArchitecture(VAEArchitecture):

    def __init__(self, input_shape, latent_dim):
        self.encoder = self.__create_encoder(input_shape, latent_dim)
        self.decoder = self.__create_decoder(latent_dim)

    def __create_encoder(self, input_shape, latent_dim):
        encoder_inputs = keras.Input(shape=input_shape)

        start_filters = 8
        blocks = 2

        x = encoder_inputs

        for i in range(0, blocks):
            x = layers.Conv2D(start_filters * (2 ** i), 3, activation='relu', padding='same',
                              kernel_initializer='he_normal')(x)
            x = layers.Conv2D(start_filters * (2 ** i), 3, activation='relu', padding='same',
                              kernel_initializer='he_normal')(x)
            x = layers.MaxPooling2D()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

        z = Sampling()([z_mean, z_log_var])

        enc = keras.Model(
            encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        return enc

    def __create_decoder(self, latent_dim):
        latent_inputs = keras.Input(shape=(latent_dim))

        x = layers.Dense(16 * 16 * 32, activation="relu")(latent_inputs)
        x = layers.Reshape((16, 16, 32))(x)

        start_filters = 8
        blocks = 2

        for i in range(0, blocks):
            mult = (2 ** (blocks - i - 1))

            x = layers.UpSampling2D()(x)
            x = layers.Conv2D(start_filters * mult, 3, activation='relu', padding='same',
                              kernel_initializer='he_normal')(x)
            x = layers.Conv2D(start_filters * mult, 3, activation='relu', padding='same',
                              kernel_initializer='he_normal')(x)

        decoder_outputs = layers.Conv2D(3, 1, activation='tanh')(x)

        dec = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        return dec
