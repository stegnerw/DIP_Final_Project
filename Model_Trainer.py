# Custom imports
from Settings import *
from TF_Dataset import getDataset, prepDataset
from Get_TF_Model import getDenseModel

# External imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import pathlib

def trainModel(model, train_ds, test_ds, model_dir, verbose=2, epochs=20, shuffle=True):
    # Fit model
    model.fit(train_ds, validation_data=test_ds, verbose=verbose, epochs=epochs, shuffle=shuffle)
    model.evaluate(test_ds)

    model.save(model_dir)

if __name__=='__main__':
    model_name = 'Dense_No_Hidden'

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

    # Generate dense model
    class_count = len(json.load(open(CLASSES_LIST)))
    hidden_layers = []
    print('Creating model')
    dense_model = getDenseModel(class_count, hidden_layers)
    dense_model.summary()

    # Train model
    model_dir = MODEL_DIR.joinpath(model_name)
    trainModel(dense_model, train_ds, test_ds, model_dir, epochs=5)
    print('Done')

