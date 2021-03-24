# TODO These imports will be redundant, if this file is imported, by a file which already uses these import statements
from typing import List
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import layers
# from enum import Enum
from custom_loss.vgg_relu_layers import VGG_ReLu_Layer


class FPL():
    """Feature Perceptual Loss
       Contains the function calculate_fp_loss
    """

    def __init__(self, input_shape, batch_size, loss_layers, beta):
        """Initializing Feature Perceptual Loss

        Args:
            input_shape (int, int, int): input shape of the image
            batch_size ([type]): [description]
            loss_layers (List[int]): nth layers of which the loss should be computed
            beta (List[int]): beta's for weighing individual loss of the layers. The length should equal the length of the loss_layers
        """
        if len(loss_layers) != len(beta):
            raise ValueError(
                '\'loss_layers\' and \'beta\' should be of the same length')
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.loss_layers = loss_layers
        self.beta = beta
        # Creating the models
        self.models = self.create_models()

    def create_models(self) -> List:
        """Creating sub-models of VGG19 for loss computation per layer

        Returns:
            List[tf.keras.Model]: List of the sub-models of VGG19
        """
        # Classes not defined? classes=<number of classes>
        # If there is some kind of 'CERTIFICATE_VERIFY_FAILED' error, go to: 'Appplications/Python 3.6', and double click 'Install Certificates.command'
        model_vgg = VGG19(input_shape=self.input_shape,
                          weights="imagenet", include_top=False)

        # Extra print statements
        # print('vgg summary:')
        # model_vgg.summary()

        # Building all models
        models = []
        for idx, layer in enumerate(self.loss_layers):
            # Take specific layers of the VGG19 model
            layers = model_vgg.layers[(
                0 if idx == 0 else self.loss_layers[idx-1].value+1):layer.value+1]  # '+1', since the Input is seen as a layer
            model = keras.Sequential(name="model_" + str(idx), layers=layers)
            model.trainable = False

            # Setting correct input shape
            sub_input_shape = (self.input_shape if idx ==
                               0 else models[idx - 1].output_shape)
            model.build(input_shape=(sub_input_shape))

            # # Extra print statements
            # for l in layers:
            #     print(l)
            # print('inpshape', sub_input_shape)
            # model.summary()

            models.append(model)

        self.l1 = models[0]
        self.l2 = models[1]
        self.l3 = models[2]
        return models

    def calculate_fp_loss(self, img1, img2):
        """Calculating feature perceptual loss of 2 images

        Args:
            models (List[tensorflow.keras.Model]): List of sub-models of VGG19
            img1 (tensorflow.python.data.ops.dataset_ops.MapDataset): first image
            img2 (tensorflow.python.data.ops.dataset_ops.MapDataset): second image

        Returns:
            float: Feature perceptual loss
        """
        l1_real = self.l1(img1)
        l1_gen = self.l1(img2)
        l2_real = self.l2(l1_real)
        l2_gen = self.l2(l1_gen)
        l3_real = self.l3(l2_real)
        l3_gen = self.l3(l2_gen)

        l1_loss = self.beta[0] * \
            tf.reduce_sum(tf.square(l1_real - l1_gen), [1, 2, 3])
        l2_loss = self.beta[1] * \
            tf.reduce_sum(tf.square(l2_real - l2_gen), [1, 2, 3])
        l3_loss = self.beta[2] * \
            tf.reduce_sum(tf.square(l3_real - l3_gen), [1, 2, 3])
        # total_loss = tf.add_n([l1_loss, l2_loss, l3_loss])

        total_loss = tf.reduce_mean(l1_loss + l2_loss + l3_loss)
        return total_loss

        pixel_loss = []
        fp_losses = []
        prediction_1 = img1
        prediction_2 = img2
        # For every model, caculate the feature perceptual loss
        for idx, model in enumerate(self.models):
            prediction_1 = model(prediction_1)
            prediction_2 = model(prediction_2)
            mse = tf.keras.losses.MeanSquaredError(reduction='auto')
            pixel_loss.append(mse(prediction_1, prediction_2))

            fp_losses.append(tf.reduce_sum(pixel_loss[idx]))

            # Extra print statements
            # print('prediction shape', prediction_1.shape)
            # print('pixel loss', pixel_loss[idx])
            # print('pixel loss shape', pixel_loss[idx].shape)
            # print('shape :', prediction_1.shape)
        # Computing total loss by element wise multiplication of pf_losses and beta
        total_loss = sum([f*b for f,
                          b in zip(fp_losses, self.beta)])
        return total_loss


if __name__ == '__main__':
    """Test the funcionality of the feature perceptual loss class
    """

    batch_size = 4
    fpl = FPL(batch_size=batch_size, input_shape=(64, 64, 3),
              loss_layers=[VGG_ReLu_Layer.ONE, VGG_ReLu_Layer.TWO, VGG_ReLu_Layer.THREE], beta=[0.5, 0.5, 0.5])

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