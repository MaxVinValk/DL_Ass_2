from keras import backend as K
# TODO These imports will be redundant, if this file is imported, by a file which already uses these import statements
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.applications.vgg19 import VGG19
from keras import Model
from tensorflow.keras import layers
from enum import Enum
# from ..util import layers


class Layers(Enum):
    """Layers with the first ReLU for feature perceptual loss

    Args:
        Enum (int): Number, indicating the block
    """
    ONE = 1
    TWO = 4
    THREE = 7
    FOUR = 12
    FIVE = 17


class FPL():
    """Feature Perceptual Loss
    """

    def __init__(self, input_shape, batch_size, loss_layers, beta):
        """[summary]

        Args:
            input_shape (int, int, int): input shape of the image
            batch_size ([type]): [description]
            loss_layers (List[int]): nth layers of which the loss should be computed
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

    def create_models(self):
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
        return models

    def calculate_fp_losses(self, img1, img2):
        """Calculating feature perceptual loss of 2 images

        Args:
            models (List[tensorflow.keras.Model]): List of sub-models of VGG19
            img1 (tensorflow.python.data.ops.dataset_ops.MapDataset): first image
            img2 (tensorflow.python.data.ops.dataset_ops.MapDataset): second image

        Returns:
            int: Feature perceptual loss
        """
        pixel_loss = []
        fp_losses = []
        # For every model, caculate the feature perceptual loss
        for idx, model in enumerate(self.models):
            prediciton_1 = model.predict(
                img1 if idx == 0 else prediciton_1, batch_size=self.batch_size)
            prediciton_2 = model.predict(
                img2 if idx == 0 else prediciton_2, batch_size=self.batch_size)
            mse = tf.keras.losses.MeanSquaredError(reduction='auto')
            # (= squared euclidean distance)
            pixel_loss.append(mse(prediciton_1, prediciton_2))
            fp_losses.append(np.sum(pixel_loss[idx]))

            # # Extra print statements
            # print('prediction shape', prediciton_1.shape)
            # print('pixel loss', pixel_loss[idx])
            # print('pixel loss shape', pixel_loss[idx].shape)
            # print('shape :', prediciton_1.shape)
        return fp_losses


if __name__ == '__main__':
    """Test the funcionality of the feature perceptual loss class
    """

    batch_size = 4
    fpl = FPL(batch_size=batch_size, input_shape=(128, 128, 3),
              loss_layers=[Layers.ONE, Layers.TWO, Layers.THREE], beta=[0.5, 0.5, 0.5])
    
    # TODO remove, this is for test purposes only.
    def load_celeba(folder, batch_size, image_size):
        # NOTE: shuffle is set to false, for test purposes (checking if presenting unshuffled data twice will result in 0 loss)
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            folder, image_size=image_size, batch_size=batch_size, shuffle=False)

        normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
        normalized_ds = train_ds.map(lambda x, y: normalization_layer(x))
        print(normalized_ds)
        return normalized_ds

    # Getting test images (created an extra dataset for this; could be changed to default, but will take long)
    img1 = load_celeba("celeba_one_image/data", batch_size, (128, 128))
    img2 = load_celeba("celeba_one_image2/data", batch_size, (128, 128))

    # Calculate the loss for a single image
    loss = fpl.calculate_fp_losses(img1, img2)
    print('fp_loss:', loss)
