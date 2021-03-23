import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import sys
import pickle

# TODO: Refactor and give a nice place somewhere
'''
TODO: Move to latent explorer

def create_image_from_z(decoder, z):
    rec = decoder(z)
    rec = rec.numpy().reshape(128, 128, 3)

    rec *= 255
    rec = rec.astype(np.uint8)

    return Image.fromarray(rec)


def create_sweep(decoder, z, dimension, min=-1.5, max=1.5, step_size=0.01):
    results = []

    for i in np.arange(min, max, step_size):
        zprime = z.copy()
        zprime[0][dimension] = i

        results.append(create_image_from_z(decoder, zprime))

    return results
'''


def load_celeba(folder, image_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(folder, image_size=image_size, batch_size=1)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: normalization_layer(x))

    return normalized_ds


def tensorshape_to_tk(img):
    imArray = img.numpy().reshape(64, 64, 3)
    imArray = imArray * 255

    imPil = Image.fromarray(imArray.astype(np.uint8))

    return ImageTk.PhotoImage(imPil)


def set_image_on_panel(frame, panel, image, side):
    if panel is None:
        panel = tk.Label(frame, image=image)
        panel.image = image
        panel.pack(side=side, padx=10, pady=10)
    else:
        panel.configure(image=image)
        panel.image = image

    return panel


def set_before_panel():
    global panelA, currentImg, topFrame

    beforeIm = tensorshape_to_tk(currentImg)
    panelA = set_image_on_panel(topFrame, panelA, beforeIm, "left")


def set_after_panel(z):
    global panelB, currentImg, topFrame, dec

    afterIm = tensorshape_to_tk(dec(z))
    panelB = set_image_on_panel(topFrame, panelB, afterIm, "right")


def refresh_panels():
    global currentImg, enc, origZ, copiedZ

    _, _, z = enc(currentImg)

    origZ = z.numpy()
    copiedZ = origZ.copy()

    set_before_panel()
    set_after_panel(origZ)


def load_next_face():
    global celeba, currentImg
    currentImg = next(iter(celeba))

    refresh_panels()
    change_dim_box()


def change_dim_box():
    global zScales, copiedZ

    for i in range(0, len(zScales)):
        zScales[i].set(copiedZ[0][i])


def change_z_box(event=None):
    global zScales, copiedZ

    for i in range(0, len(zScales)):
        copiedZ[0][i] = float(zScales[i].get())

    set_after_panel(copiedZ)


def randomize_z():
    global zScales, copiedZ

    copiedZ = np.random.normal(0, 1, (1, 100))

    for i in range(0, len(zScales)):
        zScales[i].set(copiedZ[0][i])

    set_after_panel(copiedZ)


def reset_z():
    global origZ, copiedZ, zScales
    copiedZ = origZ.copy()

    for i in range(0, len(zScales)):
        zScales[i].set(copiedZ[0][i])

    set_after_panel(copiedZ)


def load_average_vectors(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def apply_transform():
    global vectorCombo, vectorMode, avgVecs, copiedZ

    if vectorMode.get() == "add":
        copiedZ = (copiedZ + avgVecs[vectorCombo.get()])
    else:
        copiedZ = (copiedZ - avgVecs[vectorCombo.get()])

    set_after_panel(copiedZ)
    change_dim_box()


if __name__ == "__main__":
    CELEB_A_PATH = "celeba_cropped/data/"
    ENC_PATH = "enc"
    DEC_PATH = "dec"
    AVG_VECTORS_PATH = "avgDif" # "averageVectors"

    for i in range(1, len(sys.argv)):
        if (sys.argv[i] == "--celeba"):
            CELEB_A_PATH = str(sys.argv[i+1])
        elif (sys.argv[i] == "--enc"):
            ENC_PATH = str(sys.argv[i+1])
        elif (sys.argv[i] == "--dec"):
            DEC_PATH = str(sys.argv[i+1])
        elif (sys.argv[i] == "--avgVecs"):
            AVG_VECTORS_PATH = str(sys.argv[i+1])

    NETWORK_IMAGE_INPUT_SIZE = (64, 64)

    print("Loading in celeba data")

    celeba = load_celeba(CELEB_A_PATH, NETWORK_IMAGE_INPUT_SIZE)
    currentImg = None
    origZ = None
    copiedZ = None

    print("Loading in encoder, decoder")
    enc = keras.models.load_model(ENC_PATH)
    dec = keras.models.load_model(DEC_PATH)

    print("Loading in the average vectors")
    avgVecs = load_average_vectors(AVG_VECTORS_PATH)
    avgVecNames = list(avgVecs.keys())
    avgVecNames.sort()

    print("Initializing GUI")

    root = tk.Tk()
    root.title("Images")

    topFrame = tk.Frame(root)
    bottomFrame = tk.Frame(root)

    inputFrame = tk.Frame(bottomFrame)

    topFrame.pack(side="top")
    bottomFrame.pack(side="bottom")

    panelA = None
    panelB = None

    btn = tk.Button(bottomFrame, text="Next image", command=load_next_face)
    btn.pack(side="left", fill="none", expand="no", padx=10, pady=10)

    zScales = []

    root2 = tk.Tk()
    root2.title("Latent space")

    lastCol = 0
    lastRow = 0

    for i in range(0, 100):
        lbl = tk.Label(root2, text=f"dim: {i}")
        lbl.grid(column=int(i / 15) * 2, row=i % 15)

        scale = tk.Scale(root2, orient=tk.HORIZONTAL, resolution=0.01, from_=-5, to=5, command=change_z_box)
        scale.grid(column=int(i / 15) * 2 + 1, row=i % 15)

        zScales.append(scale)

    placement = 103

    resetBtn = tk.Button(root2, text="Reset Z", width=10, command=reset_z)
    randomBtn = tk.Button(root2, text="Randomize", width=10, command=randomize_z)

    vectorCombo = ttk.Combobox(root2, values=avgVecNames, state="readonly")
    vectorCombo.current(0)

    vectorMode = ttk.Combobox(root2, values=["add", "subtract"], state="readonly")
    vectorMode.current(0)

    vectorBtn = tk.Button(root2, text="Apply transform", width=10, command=apply_transform)

    resetBtn.grid(column=int(placement / 15) * 2 + 1, row=placement % 15)
    placement += 1
    randomBtn.grid(column=int(placement / 15) * 2 + 1, row=placement % 15)
    placement += 1

    vectorCombo.grid(column=int(placement / 15) * 2 + 1, row=placement % 15)
    placement+=1
    vectorMode.grid(column=int(placement / 15) * 2 + 1, row=placement % 15)
    placement+=1
    vectorBtn.grid(column=int(placement / 15) * 2 + 1, row=placement % 15)
    placement+=1

    load_next_face()

    root.mainloop()
