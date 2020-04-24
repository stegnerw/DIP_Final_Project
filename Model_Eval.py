# Make sure TF logs have correct verbosity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Custom imports
from Settings import *
from TF_Dataset import getDataset, prepDataset

# External imports
import numpy as np
import pathlib
# import subprocess
# import time

# # Ignore these functions for now, they don't work

# def waitForServer(url, wait_time=50, max_attempts=40):
    # """
    # Ping server periodically until its hosted or max_attempts

    # Parameters
        # url : str
            # String URL to ping
        # wait_time : int, optional
            # Milliseconds to wait between pings
        # max_attempts : int, optional
            # Maximum attempts to connect to the server
            # If None, it tries forever

    # Returns
        # bool
            # True if server connected
            # False if not
    # """
    # attempts = 0
    # while (not max_attempts or attempts < max_attempts):
        # attempts += 1
        # if (0 == subprocess.call(['curl', '--output', '/dev/null', '--silent', '--head', '--fail', f'{url}'])):
            # # print(f'Found server on attempt {attempts}')
            # return True
        # # print(f'Server not found on attempt {attempts}')
        # time.sleep(wait_time / 1000.0)
    # return False

# def downloadCSV(model_dir, host, port):
    # # Parse model name
    # model_name = model_dir.name

    # # Return value
    # gotFile = False

    # # Launch TensorBoard in separate pipe and wait for server to open
    # tb_dir = model_dir.joinpath(TB_DIR_NAME)
    # print(f'Starting TensorBoard server on {host}:{port}')
    # print(f'Model: {str(model_dir)}')
    # print(tb_dir.exists())
    # tb_pipe = subprocess.Popen(['tensorboard', '--logdir', f'{str(tb_dir)}', '--port', f'{port}'])
    # tb_test_url = f'{host}:{port}'
    # if waitForServer(tb_test_url, max_attempts=30, wait_time=100):
        # print(f'Connected to {tb_test_url}')
    # else:
        # print('Big oof')

    # for f_type in ['train', 'validation']:
        # out_file    = str(TRAINING_LOG_DIR.joinpath(f'{model_name}_{f_type}.csv'))
        # # csv_url     = f'{host}:{port}/data/plugin/scalars/scalars?tag=epoch_accuracy&run=TensorBoard%2F{f_type}&format=csv'
        # csv_url     = 'localhost:8080/data/plugin/scalars/scalars?tag=epoch_accuracy&run=TensorBoard%2Ftrain&format=csv'
        # # subprocess.call(['curl', '--silent', '--output', out_file, csv_url])
        # subprocess.call(['curl', csv_url])
    # # Close TensorBoard pipe
    # # tb_pipe.terminate()
    # return gotFile

def evalNetworkAccuracy(model_dir):
    network_eval = {}
    network_eval['name'] = model_dir.name
    print(f'Testing network {network_eval["name"]}')
    model = tf.keras.models.load_model(str(model_dir))

    # Load and prep datasets
    train_dataset   = prepDataset(
        getDataset(TRAIN_DIR),
        batch_size=1,
        shuffle_buffer_size=1000,
        reshuffle_each_iteration=False,
        shuffle_seed=0
    )

    test_dataset   = prepDataset(
        getDataset(TEST_DIR),
        batch_size=1,
        shuffle_buffer_size=1000,
        reshuffle_each_iteration=False,
        shuffle_seed=0
    )

    # Evaluate datasets
    # Train data
    metrics = model.evaluate(train_dataset, verbose=1)
    network_eval['train_loss'] = metrics[0]
    network_eval['train_accuracy'] = metrics[1]

    # Test data
    metrics = model.evaluate(test_dataset, verbose=1)
    network_eval['test_loss'] = metrics[0]
    network_eval['test_accuracy'] = metrics[1]

    return network_eval

def getConfusionMatrix(model_dir):
    # Lists to store pred vs label for later use
    conf_mat_vals = {
        'Preds': [],
        'Labels': []
    }

    # Load model
    print(f'Getting confusion matrix for network {model_dir.name}')
    model = tf.keras.models.load_model(str(model_dir))

    # Load test dataset
    test_dataset   = prepDataset(
        getDataset(TEST_DIR),
        batch_size=1,
        shuffle_buffer_size=1000,
        reshuffle_each_iteration=False,
        shuffle_seed=0
    )

    # Evaluate each image and append to list
    for i, data in enumerate(test_dataset):
        label = data[1]
        conf_mat_vals['Labels'].append(list(label[0]).index(True))
        pred = model.predict(data[0])
        conf_mat_vals['Preds'].append(np.argmax(pred))
        # if ((i)%100) == 0:
            # print(f'Testing {i:05d}')
            # print(f'  Label: {list(label[0]).index(True)}')
            # print(f'  Pred : {np.argmax(pred)}')

    return tf.math.confusion_matrix(
        conf_mat_vals['Labels'],
        conf_mat_vals['Preds']
    ).numpy()


if __name__ == '__main__':
    # # Ignore this stuff for now, it didn't work

    # # Constant definitions
    # HOST        = 'localhost'
    # PORT        = '8080'
    # URL         = f'{HOST}:{PORT}'

    # # Create necessary directories
    TRAINING_LOG_DIR.mkdir(exist_ok=True)

    # test_model_dir = MODEL_DIR.joinpath('Dense_0x000')
    # downloadCSV(test_model_dir, HOST, PORT)

    # Evaluate loss/accuracy of each network
    evals = []
    for m_dir in MODEL_DIR.iterdir():
        evals.append(evalNetworkAccuracy(m_dir))

    # Save results as csv
    csv_header = ['name', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']
    csv_file = []
    csv_file.append(csv_header)
    for ev in evals:
        new_row = []
        for key in csv_header:
            new_row.append(ev[key])
        csv_file.append(new_row)
    csv_file = np.asarray(csv_file)
    np.savetxt(str(TRAINING_LOG_DIR.joinpath('acc_loss.csv')), csv_file, delimiter=',', fmt='%s')

    m_dirs = [d for d in MODEL_DIR.iterdir()]
    m_dirs.sort()
    for m_dir in m_dirs:
        conf_mat = getConfusionMatrix(m_dir)
        csv_name = f'{m_dir.name}.csv'
        np.savetxt(
            str(TRAINING_LOG_DIR.joinpath(csv_name)),
            conf_mat, delimiter=',', fmt='%u'
        )

    print('Done')

