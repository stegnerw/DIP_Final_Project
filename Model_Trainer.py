# Custom imports
from Settings import *
from TF_Dataset import getDataset, prepDataset
from Get_TF_Model import getDenseModel

# External imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import pathlib

def trainModel(model, train_ds, test_ds, model_dir, verbose=2, epochs=20, shuffle=True):
    # Create tensorboard callback
    tensorboard_dir = model_dir.joinpath('TensorBoard')
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(tensorboard_dir), histogram_freq=1)

    # Fit model
    model.fit(train_ds, validation_data=test_ds, verbose=verbose, epochs=epochs, shuffle=shuffle, callbacks=[tensorboard_callback])
    model.evaluate(test_ds)

    model.save(str(model_dir))

if __name__=='__main__':
    # Configure TF memory usage
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('Found and configured GPU')
    else:
        print('Cannot find GPU')

    # Generate test and train datasets
    print('Creating train dataset')
    train_ds = getDataset(TRAIN_DIR)
    print('Creating test dataset')
    test_ds = getDataset(TEST_DIR)

    # Dataset prep parameters
    batch_size = 32
    shuffle_buffer_size = 1000
    reshuffle_each_iteration = True
    shuffle_seed = 0

    # Prep dataset for training
    train_ds = prepDataset(
        train_ds,
        batch_size=batch_size,
        shuffle_buffer_size=1000,
        reshuffle_each_iteration=reshuffle_each_iteration,
        shuffle_seed=shuffle_seed
    )
    test_ds = prepDataset(
        test_ds,
        batch_size=batch_size,
        shuffle_buffer_size=1000,
        reshuffle_each_iteration=reshuffle_each_iteration,
        shuffle_seed=shuffle_seed
    )

    # Lists for models and model names
    models = []
    model_names = []

    # Generate dense model with no hidden layers
    # Set model parameters
    class_count = len(json.load(open(CLASSES_LIST)))
    hidden_layers = []
    model_name = f'Dense_0x{0:03d}'
    # Generate model
    #print(f'Creating model {model_name}')
    models.append(getDenseModel(class_count, hidden_layers))
    model_names.append(model_name)

    # Generate dense model with hidden layers
    for hl_count in [1, 2, 3]:
        for hl_size in [32, 64, 128, 256, 512]:
            # Set model parameters
            class_count = len(json.load(open(CLASSES_LIST)))
            hidden_layers = [hl_size] * hl_count
            model_name = f'Dense_{hl_count}x{hl_size:03d}'
            # Generate model
            #print(f'Creating model {model_name}')
            models.append(getDenseModel(class_count, hidden_layers))
            model_names.append(model_name)

    # Train models
    for model, model_name in zip(models, model_names):
        model_dir = MODEL_DIR.joinpath(model_name)
        trainModel(model, train_ds, test_ds, model_dir, epochs=420, verbose=1)
        #print(f'Done with {model_name}')

