# Custom imports
from Settings import *
from TF_Dataset import getDataset, prepDataset

# External imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import subprocess
import time

def waitForServer(url, wait_time=50, max_attempts=40):
    """
    Ping server periodically until its hosted or max_attempts

    Parameters
        url : str
            String URL to ping
        wait_time : int, optional
            Milliseconds to wait between pings
        max_attempts : int, optional
            Maximum attempts to connect to the server
            If None, it tries forever

    Returns
        bool
            True if server connected
            False if not
    """
    attempts = 0
    while (not max_attempts or attempts < max_attempts):
        attempts += 1
        if (0 == subprocess.call(['curl', '--output', '/dev/null', '--silent', '--head', '--fail', f'{url}'])):
            # print(f'Found server on attempt {attempts}')
            return True
        # print(f'Server not found on attempt {attempts}')
        time.sleep(wait_time / 1000.0)
    return False

def downloadCSV(model_dir, host, port):
    # Parse model name
    model_name = model_dir.name

    # Return value
    gotFile = False

    # Launch TensorBoard in separate pipe and wait for server to open
    tb_dir = model_dir.joinpath(TB_DIR_NAME)
    print(f'Starting TensorBoard server on {host}:{port}')
    print(f'Model: {str(model_dir)}')
    print(tb_dir.exists())
    tb_pipe = subprocess.Popen(['tensorboard', '--logdir', f'{str(tb_dir)}', '--port', f'{port}'])
    tb_test_url = f'{host}:{port}'
    if waitForServer(tb_test_url, max_attempts=30, wait_time=100):
        print(f'Connected to {tb_test_url}')
    else:
        print('Big oof')

    for f_type in ['train', 'validation']:
        out_file    = str(TRAINING_LOG_DIR.joinpath(f'{model_name}_{f_type}.csv'))
        # csv_url     = f'{host}:{port}/data/plugin/scalars/scalars?tag=epoch_accuracy&run=TensorBoard%2F{f_type}&format=csv'
        csv_url     = 'localhost:8080/data/plugin/scalars/scalars?tag=epoch_accuracy&run=TensorBoard%2Ftrain&format=csv'
        # subprocess.call(['curl', '--silent', '--output', out_file, csv_url])
        subprocess.call(['curl', csv_url])
    # Close TensorBoard pipe
    # tb_pipe.terminate()
    return gotFile

if __name__ == '__main__':
    # Constant definitions
    HOST        = 'localhost'
    PORT        = '8080'
    URL         = f'{HOST}:{PORT}'

    # Create necessary directories
    TRAINING_LOG_DIR.mkdir(exist_ok=True)

    test_model_dir = MODEL_DIR.joinpath('Dense_0x000')
    downloadCSV(test_model_dir, HOST, PORT)
    print('Done')

