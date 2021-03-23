import sys
import os
import datetime

from cnn.fpl import FPL
from cnn.vgg_relu_layers import *

from data.dataset_loader import CelebA
from vae.vae import VAE

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

    fpl = FPL(
        input_shape=input_shape,
        batch_size=BATCH_SIZE,
        loss_layers=[VGG_ReLu_Layer.ONE,
                     VGG_ReLu_Layer.TWO, VGG_ReLu_Layer.THREE],
        beta=[.5, .5, .5])

    vae = VAE(input_shape=input_shape, latent_dim=latent_dim, fpl=fpl)
    celeba_data = CelebA(DATA_PATH, BATCH_SIZE, (RESIZE_HEIGHT, RESIZE_WIDTH))

    vae.train(data=celeba_data, epochs=30, log_dir=LOG_DIR)

    vae.save(EXP_DIR)