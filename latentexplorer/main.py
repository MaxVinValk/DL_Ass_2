import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
import tkinter as tk
from PIL import Image, ImageTk
import sys


def load_celeba(folder, image_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(folder, image_size=image_size, batch_size=1)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: normalization_layer(x))

    return normalized_ds


def tensorshape_to_tk(img):
    imArray = img.numpy().reshape(128, 128, 3)
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

    copiedZ = np.random.normal(0, 1, (1, 50))

    for i in range(0, len(zScales)):
        zScales[i].set(copiedZ[0][i])

    set_after_panel(copiedZ)


def reset_z():
    global origZ, copiedZ, zScales
    copiedZ = origZ.copy()

    for i in range(0, len(zScales)):
        zScales[i].set(copiedZ[0][i])

    set_after_panel(copiedZ)


if __name__ == "__main__":
    CELEB_A_PATH = "celebaSubset/img_align_celeba"
    ENC_PATH = "enc"
    DEC_PATH = "dec"

    for i in range(1, len(sys.argv)):
        if (sys.argv[i] == "--celeba"):
            CELEB_A_PATH = str(sys.argv[i+1])
        elif (sys.argv[i] == "--enc"):
            ENC_PATH = str(sys.argv[i+1])
        elif (sys.argv[i] == "--dec"):
            DEC_PATH = str(sys.argv[i+1])


    NETWORK_IMAGE_INPUT_SIZE = (128, 128)

    print("Loading in celeba data")

    celeba = load_celeba(CELEB_A_PATH, NETWORK_IMAGE_INPUT_SIZE)
    currentImg = None
    origZ = None
    copiedZ = None

    print("Loading in encoder, decoder")
    enc = keras.models.load_model(ENC_PATH)
    dec = keras.models.load_model(DEC_PATH)

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

    for i in range(0, 50):
        lbl = tk.Label(root2, text=f"dim: {i}")
        lbl.grid(column=int(i / 15) * 2, row=i % 15)

        scale = tk.Scale(root2, orient=tk.HORIZONTAL, resolution=0.01, from_=-5, to=5, command=change_z_box)
        scale.grid(column=int(i / 15) * 2 + 1, row=i % 15)

        zScales.append(scale)

    resetBtn = tk.Button(root2, text="Reset Z", width=10, command=reset_z)
    randomBtn = tk.Button(root2, text="Randomize", width=10, command=randomize_z)

    resetBtn.grid(column=int(50 / 15) * 2 + 1, row=50 % 15)
    randomBtn.grid(column=int(51 / 15) * 2 + 1, row=51 % 15)

    load_next_face()

    root.mainloop()
