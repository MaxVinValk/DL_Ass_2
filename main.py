import sys
import os
import datetime

from tensorflow import keras

from custom_loss.custom_loss_functions import PaperLoss123, PaperLoss, ReconLoss
from custom_loss.vgg_relu_layers import VGG_ReLu_Layer

from data.dataset_loader import CelebA
from vae.vae import VAE
from vae.vae_architectures import PaperArchitecture, TutorialArchitecture

from util.beta_iterator import BetaIterator

if __name__ == '__main__':

    DATA_PATH = "celeba_cropped/data"

    TIME_STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    EXP_DIR = f"Experiment_{TIME_STAMP}"
    os.mkdir(EXP_DIR)

    # LOG_DIR = f"{EXP_DIR}/logs"

    EPOCHS = 7

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "--folder":
            DATA_PATH = str(sys.argv[i + 1])
        if sys.argv[i] == "--epochs":
            EPOCHS = int(sys.argv[i + 1])

    RESIZE_HEIGHT = 64
    RESIZE_WIDTH = 64

    BATCH_SIZE = 64

    input_shape = (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
    latent_dim = 100

    data = CelebA(DATA_PATH, BATCH_SIZE, (RESIZE_HEIGHT, RESIZE_WIDTH))

    betaValues = BetaIterator(beta_sum = 150, max_ratio = 3)

    for ratios, betas in iter(betaValues):

        ratioString = str(ratios).replace("[", "").replace("]", "").replace(",", "").replace(" ", "_")

        runFolderName = f"{EXP_DIR}/{ratioString}"
        runFolderLogs = f"{EXP_DIR}/{ratioString}/logs"

        os.mkdir(runFolderName)


        architecture = PaperArchitecture(input_shape, latent_dim)

        # Swap this out for any of the other loss functions in custom_loss.custom_loss_functions
        loss_layers = [VGG_ReLu_Layer.ONE, VGG_ReLu_Layer.TWO, VGG_ReLu_Layer.THREE]

        loss_function = PaperLoss(input_shape, BATCH_SIZE, loss_layers, 1.0, betas)
        # loss_function = PaperLoss123(input_shape, BATCH_SIZE)

        # The optimizer calls the schedule once per train_step = 1 batch,
        # we only want to change the learning rate after a batch, and we want the
        # learning rate to remain the same in between these adjustments
        learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0005,
            decay_steps=data.get_batches_per_epoch(),
            decay_rate=0.5,
            staircase=True
        )

        vae = VAE(architecture)

        vae.train(data=data, epochs=EPOCHS, custom_loss=loss_function, learning_rate=learning_rate_schedule,
                  log_dir=runFolderLogs)

        vae.save(runFolderName)

        with open(f"{runFolderName}/info.txt", "w") as f:
            f.write("GENERAL:\n\n" + f"Epochs: {EPOCHS}\n\n")
            f.write("DATA:\n\n" + str(data) + "\n\n")
            f.write("ARCHITECTURE:\n\n" + str(architecture) + "\n\n")
            f.write("LOSS FUNCTION:\n\n" + str(loss_function))
