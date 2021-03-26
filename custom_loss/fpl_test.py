from typing import List
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import layers
# from enum import Enum
from custom_loss.vgg_relu_layers import VGG_ReLu_Layer
import sys

from custom_loss.fpl import *

if __name__ == '__main__':
    """Test the funcionality of the feature perceptual loss class
    """

    batch_size = 4
    fpl = FPL(batch_size=batch_size, input_shape=(64, 64, 3),
              loss_layers=[VGG_ReLu_Layer.ONE, VGG_ReLu_Layer.TWO, VGG_ReLu_Layer.THREE], beta=[100, 100, 100])

    # TODO remove, this is for test purposes only.
    def load_celeba(folder, batch_size, image_size):
        # NOTE: shuffle is set to false, for test purposes (checking if presenting unshuffled data twice will result in 0 loss)
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            folder, image_size=image_size, batch_size=batch_size, shuffle=False)

        normalization_layer = layers.experimental.preprocessing.Rescaling(
            1. / 255)
        normalized_ds = train_ds.map(lambda x, y: normalization_layer(x))
        print(normalized_ds)
        return normalized_ds

    # Getting test images (created an extra dataset for this; could be changed to default, but will take long)
    img1 = load_celeba("celeba_one_image/data", batch_size, (64, 64))
    img2 = load_celeba("celeba_one_image2/data", batch_size, (64, 64))

    # Calculate the loss for a single image
    loss = fpl.calculate_fp_loss(img1, img2)
    print('fp_loss:', loss)
