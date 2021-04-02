import os
import pickle
import tensorflow as tf # To have the warnings appear early
from tensorflow.python.summary.summary_iterator import summary_iterator
from operator import itemgetter


def name_and_val_from_string(string):
    name = str(string[string.find(":") + 3:string.find("\n") - 1]).strip()
    val = float(string[string.find("simple_value") + 14: -1].strip())

    return name, val


def get_v2_file(folder):
    for f in os.scandir(folder):
        if not f.is_dir() and f.name.endswith(".v2"):
            return f.name
    print("No V2 file found...")
    exit(1)


def file_to_data(filepath):
    valuesFound = {}
    maxEpoch = 0

    for s in summary_iterator(filepath):
        if s.step > maxEpoch:
            maxEpoch = s.step

    for s in summary_iterator(filepath):

        step = s.step
        data = str(s.summary.value)

        if "epoch_kl_loss" not in data and "epoch_fp_loss" not in data:
            continue

        name, val = name_and_val_from_string(str(s.summary.value))

        if name not in valuesFound.keys():
            valuesFound[name] = [0] * (maxEpoch + 1)  # Epochs 0-index

        valuesFound[name][step] = val

    return valuesFound


def collect_results(root):
    results = {}

    for f in os.scandir(root):
        if f.is_dir():
            v2 = get_v2_file(f"{root}/{f.name}/logs/train")
            results[f.name] = file_to_data(f"{root}/{f.name}/logs/train/{v2}")

    return results

def get_lowest_final(data):

    results = []

    for key, value in data.items():
        total_loss = value["epoch_kl_loss"][-1] + value["epoch_fp_loss"][-1]
        results.append([key, total_loss, value["epoch_kl_loss"][-1], value["epoch_fp_loss"][-1]])

    sortedResults = sorted(results, key=itemgetter(1))

    return sortedResults


if __name__ == '__main__':
    ROOT_EXP_FOLDER = "TMP"

    results = collect_results(ROOT_EXP_FOLDER)

    with open("results", "wb") as f:
        pickle.dump(results, f)
