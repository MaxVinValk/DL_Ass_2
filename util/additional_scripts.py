import os
import pandas as pd
import numpy as np
from PIL import Image
import datetime


def create_dirs(folder_path):
    """

    Args:
        folder_path: A string representing the file path of folder(s) that need to be created.

    Returns: Nothing

    """
    # Create target folder if it does not exist already:
    foldersNeeded = folder_path.split("/")
    for i in range(len(foldersNeeded)):
        current_path = ""
        for j in range(i + 1):
            current_path += foldersNeeded[j] + "/"

        if not os.path.exists(current_path):
            os.mkdir(current_path)


def create_image_from_z(decoder, z):
    """

    Args:
        decoder: The VAE decoder
        z: latent space values to turn into an image

    Returns: a PIL.Image object

    """
    rec = decoder(z)
    rec = rec.numpy().reshape(64, 64, 3)

    rec = np.clip(rec * 255, 0, 255)
    rec = rec.astype(np.uint8)

    return Image.fromarray(rec)


def create_transition(enc, dec, img_1, img_2, num_steps):
    """

    Args:
        enc: The VAE encoder
        dec: The VAE decoder
        img_1: Image loaded with load_image_nn
        img_2: Image loaded with load_image_nn
        num_steps: The amount of steps to transition from one to the other

    Returns: Nothing

    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outputDir = f"Transition_{timestamp}"
    os.mkdir(outputDir)

    _, _, z1 = enc(img_1)
    _, _, z2 = enc(img_2)

    stepSize = (z2 - z1)/num_steps

    for i in range(0, num_steps):
        transition_image = create_image_from_z(dec, z1)
        transition_image.save(f"{outputDir}/{i}.png")
        z1 += stepSize

    final_image = create_image_from_z(dec, z1)
    final_image.save(f"{outputDir}/{num_steps}.png")


def load_image_nn(path):
    """

    Args:
        path: Path to the image one wishes to load

    Returns: An array representation of the image ready for use in the VAE encoder

    """
    img = Image.open(path)
    imgArr = np.array(img) / 255.0
    imgResized = imgArr.reshape(1, imgArr.shape[0], imgArr.shape[1], imgArr.shape[2])

    return imgResized


def create_celeba_feature_averages(img_folder, feature_file, image_size, enc):
    """

    Args:
        img_folder: The direct folder containing the images.
        feature_file: The CSV file which lists which features are present in which images
        image_size: The size of the image that we accept. A 2D tuple
        enc: The encoder of the VAE

    Returns:
        featureVectors: The averaged latent space vector per class
        featureCount:   The amount of occurrences per class that went into creating the average

    """

    attributeData = pd.read_csv(feature_file)
    attributeNames = list(attributeData.columns)
    attributeNames.remove('image_id')

    featureCount = {}
    featureVectors = {}

    for attribute in attributeNames:
        featureCount[attribute] = 0
        featureVectors[attribute] = np.zeros(shape=(1, 100))

    ctr = 0

    for filename in os.listdir(img_folder):
        if filename.endswith('.jpg'):

            ctr += 1

            if ctr % 1000 == 0:
                print(f"{ctr} images processed")

            img = load_image_nn(f"{img_folder}/{filename}")

            _, _, z = enc(img)
            z = z.numpy()

            fileNumber = int(filename[:-4])
            featuresForImage = attributeData.iloc[fileNumber - 1]

            for attribute in attributeNames:
                if featuresForImage[attribute] == 1:
                    featureVectors[attribute] += z
                    featureCount[attribute] += 1

    for attribute in attributeNames:
        featureVectors[attribute] /= featureCount[attribute]

    return featureVectors, featureCount




# TODO: Turn into a function, maybe?
'''
def create_sweep(decoder, z, dimension, min=-1.5, max=1.5, step_size=0.01):
    results = []

    for i in np.arange(min, max, step_size):
        zprime = z.copy()
        zprime[0][dimension] = i

        results.append(create_image_from_z(decoder, zprime))

    return results
'''

