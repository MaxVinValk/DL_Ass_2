import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import pad
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from cnn.fpl import FPL
from cnn.vgg_relu_layers import *
# Reshape size
RESIZE_HEIGHT = 128
RESIZE_WIDTH = 128

# https://keras.io/examples/generative/vae/

# https://fairyonice.github.io/My-first-GAN-using-CelebA-data.html for CELEBA loading

# https://towardsdatascience.com/variational-autoencoders-vaes-for-dummies-step-by-step-tutorial-69e6d1c9d8e9

# https://towardsdatascience.com/generating-new-faces-with-variational-autoencoders-d13cfcb5f0a8

# Replication padding:
# https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras/#replication-padding

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
    def __init__(self, encoder, decoder, fpl: FPL, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.fpl = fpl
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
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
                tf.reduce_sum(keras.losses.binary_crossentropy(
                    data, reconstruction), axis=(1, 2))
            )
            fpl_loss = self.fpl.calculate_fp_loss(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            # total_loss = reconstruction_loss + kl_loss
            total_loss = kl_loss + fpl_loss

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

'''
  2D Replication Padding
  Attributes:
    - padding: (padding_width, padding_height) tuple
'''


class ReplicationPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReplicationPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return pad(input_tensor, [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]], 'SYMMETRIC')


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
    # x = layers.Dense(128, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = Sampling()([z_mean, z_log_var])

    enc = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    print(enc.summary())
    return enc, pre_flatten_shape


def create_decoder(latent_dim, pre_flatten_shape):
    latent_inputs = keras.Input(shape=(latent_dim))

    x = layers.Dense(np.prod(pre_flatten_shape))(latent_inputs)
    x = layers.Reshape(pre_flatten_shape)(x)

    start_filters = 8
    blocks = 3

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
    decoder_outputs = ReplicationPadding2D(padding=(1, 1))(x)

    dec = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    print(dec.summary())
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
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        folder, image_size=image_size, batch_size=batch_size)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: normalization_layer(x))

    return normalized_ds


'''
    folder: The folder containing the images. Note that the other functions usually take the folder that
            contains the folder which has the images
            
    featureFile: The CSV file which lists which features are present in which images
    
    imageSize: The size of the image that we accept. A 2D tuple
    
    enc: The loaded encoder
'''


def create_celeba_feature_averages(folder, featureFile, imageSize, enc):
    attributeData = pd.read_csv(featureFile)
    attributeNames = list(attributeData.columns)
    attributeNames.remove('image_id')

    featureCount = {}
    featureVectors = {}

    for attribute in attributeNames:
        featureCount[attribute] = 0
        featureVectors[attribute] = np.zeros(shape=(1, 50))

    ctr = 0

    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):

            ctr += 1

            if (ctr % 1000 == 0):
                print(f"{ctr} images processed")

            img = np.array(Image.open(
                f"{folder}/{filename}").resize(imageSize))
            img = (img / 255)
            # img = img.reshape(1, 128, 128, 3)

            _, _, z = enc(img)
            z = z.numpy()

            fileNumber = int(filename[:-4])
            featuresForImage = attributeData.iloc[fileNumber - 1]

            for attribute in attributeNames:
                if featuresForImage[attribute] == 1:
                    featureVectors[attribute] += z
                    featureCount[attribute] += 1

    for attribute in attributeNames:
        featureVectors[attribute] /= featureCount[attribute]

    return featureVectors, featureCount


if __name__ == '__main__':

    # General setup for all other modes

    DATA_PATH = "celeba_vsmall/data"
    RUN_MODE = "train"

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "--folder":
            DATA_PATH = str(sys.argv[i + 1])

    input_shape = (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
    latent_dim = 100

    BATCH_SIZE = 128

    if (RUN_MODE == "train"):
        fpl = FPL(
            input_shape=input_shape,
            batch_size=BATCH_SIZE,
            loss_layers=[VGG_ReLu_Layer.ONE,
                         VGG_ReLu_Layer.TWO, VGG_ReLu_Layer.THREE],
            beta=[1., 1., 1.])
        encoder, pre_flatten_shape = create_encoder(
            input_shape=input_shape, latent_dim=latent_dim)
        decoder = create_decoder(
            latent_dim=latent_dim, pre_flatten_shape=pre_flatten_shape)

        vae = VAE(encoder, decoder, fpl)

        data = load_celeba(DATA_PATH, BATCH_SIZE,
                           (RESIZE_HEIGHT, RESIZE_WIDTH))

        train_VAE(vae, data, epochs=2, batch_size=BATCH_SIZE)

        # encoder.save("enc/")
        # decoder.save("dec/")
