import sys
import os
from shutil import copyfile

SOURCE_FOLDER = "celeba/data/img_align_celeba"
TARGET_FOLDER = "celeba_vsmall/data/img_align_celeba"
HOWMANY = 500

copied = 0

oneP = int(HOWMANY / 100)

for filename in os.listdir(SOURCE_FOLDER):

    if (copied % oneP == 0):
        print(f"{int(copied / oneP)}% Done")

    if (copied >= HOWMANY):
        break

    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        copyfile(f"{SOURCE_FOLDER}/{filename}", f"{TARGET_FOLDER}/{filename}")
        copied += 1
