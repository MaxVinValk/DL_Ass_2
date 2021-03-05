import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image


# https://keras.io/examples/generative/vae/

# https://fairyonice.github.io/My-first-GAN-using-CelebA-data.html for CELEBA loading

# https://towardsdatascience.com/variational-autoencoders-vaes-for-dummies-step-by-step-tutorial-69e6d1c9d8e9

# https://towardsdatascience.com/generating-new-faces-with-variational-autoencoders-d13cfcb5f0a8

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit"""

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)

            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)

            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()
            }

    '''
    def predict_image(self, image):
        # Allows passing in an unedited image
        reshaped = image

        if len(image.shape) != 4:
            reshaped = image.reshape(-1, 28, 28, 1)

        _, _, z = self.encoder(reshaped)
        rec = self.decoder(z)

        return rec.numpy()
    '''


'''
def show_prediction(image):
    reshaped = image

    if len(image.shape) == 4:
        reshaped = image.reshape(28, 28)

    img = Image.fromarray(reshaped * 255)
    img.show()
'''


def get_MNIST():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

    return mnist_digits


def train_VAE(vae, data, epochs, batch_size):
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(data, epochs=epochs, batch_size=batch_size)


def create_encoder(input_shape, latent_dim):
    encoder_inputs = keras.Input(shape=input_shape)

    start_filters = 8
    blocks = 3

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

    enc = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    return enc


def create_decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim))

    x = layers.Dense(16 * 16 * 32, activation="relu")(latent_inputs)
    x = layers.Reshape((16, 16, 32))(x)

    start_filters = 8
    blocks = 3

    for i in range(0, blocks):
        mult = (2 ** (blocks - i - 1))

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(start_filters * mult, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = layers.Conv2D(start_filters * mult, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

    # decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

    decoder_outputs = layers.Conv2D(3, 1, activation='tanh')(x)

    dec = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return dec


def create_image_from_z(decoder, z):
    rec = decoder(z)
    rec = rec.numpy().reshape(128, 128, 3)

    rec *= 255
    rec = rec.astype(np.uint8)

    return Image.fromarray(rec)


def create_sweep(decoder, z, dimension, min=-1.5, max=1.5, step_size=0.01):
    results = []

    for i in np.arange(min, max, step_size):
        zprime = z.copy()
        zprime[0][dimension] = i

        results.append(create_image_from_z(decoder, zprime))

    return results


# https://www.tensorflow.org/tutorials/load_data/images Partially for normalization
def load_celeba(folder, batch_size, image_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(folder, image_size=image_size, batch_size=batch_size)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: normalization_layer(x))

    return normalized_ds


if __name__ == '__main__':

    DATA_PATH = "celeba/data"

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "--folder":
            DATA_PATH = str(sys.argv[i + 1])

    # Reshape size
    RESIZE_HEIGHT = 128
    RESIZE_WIDTH = 128

    input_shape = (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
    latent_dim = 50

    BATCH_SIZE = 128

    encoder = create_encoder(input_shape=input_shape, latent_dim=latent_dim)
    decoder = create_decoder(latent_dim=latent_dim)

    vae = VAE(encoder, decoder)

    data = load_celeba(DATA_PATH, BATCH_SIZE, (RESIZE_HEIGHT, RESIZE_WIDTH))

    train_VAE(vae, data, epochs=20, batch_size=BATCH_SIZE)

    encoder.save("enc/")
    decoder.save("dec/")
