import os
from abc import ABC, abstractmethod

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

'''
    Abstract class that is used so that data can be more flexibly swapped out later. If needed, we can 
    add more attributes
'''
class Data(ABC):

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_batch_size(self):
        pass

    @abstractmethod
    def get_num_files(self):
        pass

    @abstractmethod
    def get_batches_per_epoch(self):
        pass

    @abstractmethod
    def __str__(self):
        return "Default data __str__. Change this!"


class CelebA(Data):
    def __init__(self, folder, batch_size, image_size):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            folder, image_size=image_size, batch_size=batch_size)

        self.num_files = 0

        for sub in os.listdir(folder):
            if os.path.isdir(f"{folder}/{sub}"):
                self.num_files += len(os.listdir(f"{folder}/{sub}"))

        normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
        normalized_ds = train_ds.map(lambda x, y: normalization_layer(x))

        self.batch_size = batch_size
        self.dataset = normalized_ds

    def get_dataset(self):
        return self.dataset

    def get_batch_size(self):
        return self.batch_size

    def get_num_files(self):
        return self.num_files

    def get_batches_per_epoch(self):
        return int(np.ceil(self.num_files / self.batch_size))

    def __str__(self):
        return "CelebA dataset with:\n" + \
                f"Num files: {self.num_files}\n" + \
                f"Batch_size: {self.batch_size}\n"
