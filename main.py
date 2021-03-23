import sys
import os
import datetime

from tensorflow import keras

from custom_loss.custom_loss_functions import PaperLoss123, Loss123, ReconLoss

from data.dataset_loader import CelebA
from vae.vae import VAE
from vae.vae_architectures import PaperArchitecture, TutorialArchitecture

if __name__ == '__main__':

    DATA_PATH = "celeba_vsmall/data"

    TIME_STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    EXP_DIR = f"Experiment_{TIME_STAMP}"
    os.mkdir(EXP_DIR)

    LOG_DIR = f"{EXP_DIR}/logs"

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "--folder":
            DATA_PATH = str(sys.argv[i + 1])

    RESIZE_HEIGHT = 64
    RESIZE_WIDTH = 64

    BATCH_SIZE = 64

    input_shape = (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
    latent_dim = 100

    celeba_data = CelebA(DATA_PATH, BATCH_SIZE, (RESIZE_HEIGHT, RESIZE_WIDTH))

    architecture = TutorialArchitecture(input_shape, latent_dim) #PaperArchitecture(input_shape, latent_dim)

    # Swap this out for any of the other loss functions in custom_loss.custom_loss_functions
    loss_function = ReconLoss() #PaperLoss123(input_shape, BATCH_SIZE)

    # The optimizer calls the schedule once per train_step = 1 batch,
    # we only want to change the learning rate after a batch, and we want the
    # learning rate to remain the same in between these adjustments
    learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0005,
        decay_steps=celeba_data.get_batches_per_epoch(),
        decay_rate=0.5,
        staircase=True
    )

    vae = VAE(architecture)

    vae.train(data=celeba_data, epochs=30, custom_loss=loss_function, learning_rate=learning_rate_schedule,
              log_dir=LOG_DIR)

    vae.save(EXP_DIR)
