import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import pickle
from tqdm import tqdm

class ImgIterator:

    def __init__(self, images_path, percentage, preload):
        self.images_path = images_path
        self.image_names = sorted(os.listdir(images_path))

        self.num_images = int((percentage/100.0) * len(self.image_names))
        self.image_names = self.image_names[:self.num_images]

        self.preload = preload

        if self.preload:
            print(f"\033[1;32;47m Preloading images \033[m")
            self.images = []
            for img in tqdm(self.image_names):
                if img.endswith(".jpg"):
                    self.images.append(load_image_nn(f"{self.images_path}/{img}"))

    def __iter__(self):
        self.idx = 0

        return self

    def __next__(self):

        if (self.idx >= len(self.image_names)):
            raise StopIteration

        if self.preload:
            img = self.images[self.idx]
        else:
            img = load_image_nn(f"{self.images_path}/{self.image_names[self.idx]}")

        self.idx += 1
        return img

def comp_mse(img_1, img_2):
    return np.square(img_1 - img_2).mean()

def load_image_nn(path):
    img = Image.open(path)
    imgArr = np.array(img) / 255.0
    imgResized = imgArr.reshape(1, imgArr.shape[0], imgArr.shape[1], imgArr.shape[2])

    return imgResized

def reconstruct(enc, dec, img):
    _, _, z = enc(img)
    return dec(z)

def load_celeba(folder, image_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(folder, image_size=image_size, batch_size=1)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: normalization_layer(x))

    return normalized_ds

def calc_avgs(model_path, data):

    enc = keras.models.load_model(f"{model_path}/enc")
    dec = keras.models.load_model(f"{model_path}/dec")

    error = 0

    for img in tqdm(data, total=data.num_images):
        reconstructed = reconstruct(enc, dec, img)
        error += comp_mse(img, reconstructed)

    error /= data.num_images

    return error

if __name__ == "__main__":
    ROOT_PATH = "Experiment_20210401-210549"
    DATA_PATH = "celeba_cropped_new/img_align_celeba/img_align_celeba"
    PERCENTAGE = 5
    PRELOAD = True  # Preload images? Can cause the program to crash if there are
                    # many images to be loaded

    res = {}


    models = os.listdir(ROOT_PATH)

    data = ImgIterator(DATA_PATH, PERCENTAGE, PRELOAD)


    for index, model in enumerate(models):
        print(f"\033[31;47;1m Model {index} out of {len(models)} \033[m")
        res[model] = calc_avgs(f"{ROOT_PATH}/{model}", data)

    with open("results", "wb") as f:
        pickle.dump(res, f)
