# Custom imports
from Settings import *

# External imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json

def getDenseModel(class_count, hidden_layer_sizes, activation='relu'):
    """Generates dense model with given parameters

    Creates a Model object to train on the data.
    Input layer is always Flatten with dimensions given
    by the image dimensions.
    Output layer is always Dense with the number of outputs
    equal to the number of classes.

    Parameters
    ----------
    classes : int
        Number of output classes
    hidden_layer_sizes : list of int
        Number of neurons in each hidden layer
    activation : str, optional
        Activation function for hidden layers

    Returns
    -------
    tf.keras.Model
        Compiled model object generated by given parameters.
    """
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    for layer_size in hidden_layer_sizes:
        model.add(keras.layers.Dense(layer_size, activation=activation))
    model.add(keras.layers.Dense(class_count, activation='softmax'))
    model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
    return model

if __name__=='__main__':
    class_count = len(json.load(open(CLASSES_LIST)))
    print(f'Class count: {class_count}')
    model = getDenseModel(class_count, [1337, 69420])
    model.summary()
    print('Done')
