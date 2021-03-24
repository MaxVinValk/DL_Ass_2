import tensorflow as tf
from tensorflow import keras

from custom_loss.custom_loss_functions import CustomLoss
from vae.vae_architectures import VAEArchitecture


class VAE(keras.Model):
    def __init__(self, architecture: VAEArchitecture, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.architecture = architecture
        self.cl = None

    @property
    def metrics(self):
        return self.cl.get_metrics()

    def train_step(self, data):
        with tf.GradientTape() as tape:

            loss = self.cl.calculate_loss(data, self.architecture)

            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            return self.cl.get_loss_trackers()

    def train(self, data, epochs, custom_loss: CustomLoss, learning_rate, log_dir):
        self.cl = custom_loss

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)

        self.compile(optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate))

        self.fit(data.get_dataset(), epochs=epochs,
                 batch_size=data.get_batch_size(), callbacks=[tensorboard_callback])

    def save(self, folder):
        self.architecture.save(folder)
