import pickle
import itertools
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import imagehash
import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path 
from data.dataset_loader import CelebA

def get_best_ratios(file_name):
    losses = []
    with(open(file_name, "rb")) as openfile:
        while True:
            try:
                losses.append(pickle.load(openfile))
            except EOFError:
                break
        
        losses = {k: v for k, v in sorted(losses[0].items(), key=lambda item: item[1])}
        
        best_losses = list(itertools.islice(losses.items(), 5))
        print(str(best_losses))

def save_rand_img(num):
    folder = "random_img/"
    file_name = "random"
    rand = np.random.normal(0, 1, (1, 100))

    z = decode_img(rand)

    save_img(num, z, folder, file_name)

def encode_img(img):
    ENC_PATH = "enc_5_1_1"
    
    enc = keras.models.load_model(ENC_PATH)
    _, _, latent_vector = enc(img)

    return latent_vector

def decode_img(latent_vector):
    DEC_PATH = "dec_5_1_1"
    dec = keras.models.load_model(DEC_PATH)
    recon = dec(latent_vector)
    return recon

def save_img(num, recon, folder, file_name):
    imArray = recon.numpy().reshape(64, 64, 3)
    imArray = np.clip(imArray * 255, 0, 255)

    imPil = Image.fromarray(imArray.astype(np.uint8))
    imPil.save(folder + file_name + "_" + str(num) + ".jpg", "JPEG")

def load_celeba(folder, image_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(folder, image_size=image_size, batch_size=1, shuffle=True)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: normalization_layer(x))

    return normalized_ds

def reconstruct_images():
    img_dir = "celeba_cropped_new/img_subset/"

    folder = "reconstructions/1_1_5/"
    file_name = "img"

    NETWORK_IMAGE_INPUT_SIZE = (64, 64)
    print("Loading in celeba data")
    celeba = load_celeba(img_dir, NETWORK_IMAGE_INPUT_SIZE)
    setA = next(iter(celeba))    

    for i in range(0,10):
        img = setA[i]
        img = img.numpy().reshape(1, 64, 64, 3)
        latent_vector = encode_img(img)
        recon = decode_img(latent_vector)
        save_img(i, recon, folder, file_name)

def load_average_vectors(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def apply_transform(z=None, addition=True):
    AVG_VECTORS_PATH = r"latentexplorer/fv_5_1_1"

    model_ratio = r"5_1_1/"

    attribute = r"Eyeglasses"

    if addition:
        add_sub = r"Addition/"
    else:
        add_sub = r"Subtraction/"

    target_folder = r"/Users/julianbruinsma/stack/School/Master/Year 1/Deep Learning/DL_Ass_2/results/attributes/" + model_ratio + attribute + "/" + add_sub
    print('Target folder is', target_folder)
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    avgVecs = load_average_vectors(AVG_VECTORS_PATH)
    alpha = 0.1
    print(list(avgVecs.keys()))
    while alpha <= 1:
        if addition:
            z = (z + (alpha*avgVecs[attribute]))
        else:
            z = (z - (alpha*avgVecs[attribute]))
        img = decode_img(z)
        save_img(num=alpha, recon=img, folder=target_folder, file_name=attribute)
        alpha += 0.1
        alpha = round(alpha, 1)

if __name__ == "__main__":
    img_dir = "celeba_cropped_new/img_subset/"
    RESIZE_HEIGHT = 64
    RESIZE_WIDTH = 64

    data = CelebA(i, BATCH_SIZE, (RESIZE_HEIGHT, RESIZE_WIDTH))

    NETWORK_IMAGE_INPUT_SIZE = (64, 64)

    celeba = load_celeba(img_dir, NETWORK_IMAGE_INPUT_SIZE)
    print(type(celeba))
    setA = next(iter(celeba))

    latent_vector = encode_img(setA)
    # latent_vector = latent_vector.numpy().reshape(1, 64, 64, 3)
    
    apply_transform(latent_vector, addition=False)
    
            
