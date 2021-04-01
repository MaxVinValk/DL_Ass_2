import sys
import os
from PIL import Image
import pandas as pd

# Points to the folder holding the images itself
SOURCE_FOLDER = "celeba/data/img_align_celeba"

# The folder in which the output is put
TARGET_FOLDER = "celeba_cropped_new/data/img_align_celeba"

# The CSV file specifying the bounding box locations. Can be found on kaggle
# https://www.kaggle.com/jessicali9530/celeba-dataset?select=list_bbox_celeba.csv
BBOX_FILE = "celeba/list_bbox_celeba.csv"

RESCALE = (64, 64)

#Technically, there are 202599 files, but any number over that will work too
HOWMANY = 202599


# Given the dataframe and the name of a file, returns the bounding box as a tuple
def getbb(df, name):
    row = df.loc[df['image_id'] == name]
    x_1 = int(row.x_1)
    y_1 = int(row.y_1)
    width = int(row.width)
    height = int(row.height)

    return (x_1, y_1, x_1 + width, y_1 + height)



#Create target folder if it does not exist already:
foldersNeeded = TARGET_FOLDER.split("/")
for i in range(len(foldersNeeded)):
    folderPath = ""
    for j in range(i+1):
        folderPath += foldersNeeded[j] + "/"

    if not os.path.exists(folderPath):
        os.mkdir(folderPath)



copied = 0
oneP = int(HOWMANY / 100)

#bbdf = pd.read_csv(BBOX_FILE)


for filename in os.listdir(SOURCE_FOLDER):

    if (copied % oneP == 0):
        print(f"{int(copied / oneP)}% Done")

    if (copied >= HOWMANY):
        break

    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        im = Image.open(f"{SOURCE_FOLDER}/{filename}")


        #crop_box = getbb(bbdf, filename)

        #im = im.crop(crop_box)

        im = im.resize(RESCALE)

        im.save(f"{TARGET_FOLDER}/{filename}")

        copied += 1
