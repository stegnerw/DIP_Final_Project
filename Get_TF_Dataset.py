# Custom constant definitions
from Settings import *

# External imports
import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import json

# Set autotune
AUTOTUNE = tf.data.experimental.AUTOTUNE

def writeClassNamesJson(data_dir):
    class_names = [c.parts[-1] for c in data_dir.iterdir() if pathlib.Path.is_dir(c)]
    class_names.sort()
    json.dump(class_names, open(CLASSES_LIST, 'w'))

def get_label(file_path):
    class_names = json.load(open(CLASSES_LIST))
    return tf.strings.split(file_path, os.path.sep)[-2] == class_names

def decode_data(data):
    # Convert raw data into gray-scale image
    img = tf.io.decode_image(data, channels=1)
    # Convert image to floats on [0,1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize and return image
    return img #tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
    label = get_label(file_path)
    data = tf.io.read_file(file_path)
    data = decode_data(data)
    return data, label

def getDataset(data_dir, file_ext='.png'):
    """Generate and return tf.data.Dataset object.

    Parameters
    ----------
    data_dir : pathlib.Path
        Root directory of dataset partition
    file_ext : str, optional
        File extension of the data files

    Returns
    -------
    tf.data.Dataset
        TensorFlow Dataset object representing the files in data_dir
    """
    if not pathlib.Path.is_file(CLASSES_LIST):
        writeClassNamesJson(data_dir)
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return labeled_ds

if __name__ == '__main__':
    writeClassNamesJson(TEST_DIR)
    ds = getDataset(TEST_DIR)

    print('Done generating dataset.')

