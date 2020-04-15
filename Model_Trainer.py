# Custom imports
from Settings import *
from Get_TF_Dataset import getDataset
from Get_TF_Model import getDenseModel

# External imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import pathlib

def trainModel(model, train_ds, test_ds, model_dir, verbose=2, epochs=20, shuffle=True):
    model.fit(train_ds, validation_data=test_ds, verbose=verbose, epochs=epochs, shuffle=shuffle)
    model.evaluate(test_ds)

if __name__=='__main__':
    model_name = 'Dense_No_Hidden'

    # Generate test and train datasets
    print('Creating train dataset')
    train_ds = getDataset(TRAIN_DIR)
    print('Creating test dataset')
    test_ds = getDataset(TEST_DIR)

    # Generate dense model
    class_count = len(json.load(open(CLASSES_LIST)))
    hidden_layers = []
    print('Creating model')
    dense_model = getDenseModel(class_count, hidden_layers)
    dense_model.summary()

    # Train model
    model_dir = MODEL_DIR.joinpath(model_name)
    trainModel(dense_model, train_ds.batch(32), test_ds.batch(32), model_dir, epochs=5)
    print('Done')

