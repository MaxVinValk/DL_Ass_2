# TODO These imports will be redundant, if this file is imported, by a file which already uses these import statements
from typing import List
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import layers
# from enum import Enum
from custom_loss.vgg_relu_layers import VGG_ReLu_Layer
import sys


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
        self.beta = [tf.constant(float(b)) for b in beta]
        # Creating the models
        self.models, self.div_facs = self.create_models()

    def create_models(self):
        """Creating sub-models of VGG19 for loss computation per layer

        Returns:
            List[tf.keras.Model]: List of the sub-models of VGG19
            List[div_facs]: Tensors that give the normalising factor for each layer
        """

        models = []
        div_facs = []

        vgg = VGG19(input_shape=self.input_shape,
                          weights="imagenet", include_top=False)

        for layer in self.loss_layers:
            model = keras.Model(inputs=vgg.inputs,
                                outputs=vgg.layers[layer].output, trainable=False)
            div_fac = self.get_div_fac(vgg.layers[layer])

            models.append(model)
            div_facs.append(div_fac)

        return models, div_facs


    def get_div_fac(self, layer):
        os = layer.output_shape
        fv = [float(x) for x in layer.output_shape[1:]]

        return tf.multiply(float(tf.constant(2)), tf.multiply(fv[0], tf.multiply(fv[1], fv[2])))

    def calculate_fp_loss(self, img1, img2):
        """Calculating feature perceptual loss of 2 images

        Args:
            models (List[tensorflow.keras.Model]): List of sub-models of VGG19
            img1 (tensorflow.python.data.ops.dataset_ops.MapDataset): first image
            img2 (tensorflow.python.data.ops.dataset_ops.MapDataset): second image

        Returns:
            float: Feature perceptual loss
        """

        losses = []

        for model, div_fac, beta in zip(self.models, self.div_facs, self.beta):
            activation_real = model(img1)
            activation_gen = model(img2)

            loss = tf.reduce_sum(tf.square(tf.subtract(activation_real, activation_gen)), [1, 2, 3])
            dividedLoss = tf.divide(loss, div_fac)

            betaScaledLoss = tf.multiply(beta, dividedLoss)

            losses.append(betaScaledLoss)

        total_loss = tf.add_n(losses)
        return total_loss
