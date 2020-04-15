import pathlib

# Directory locations
BASE_DIR    = pathlib.Path(__file__).resolve().parent
TRAIN_DIR   = BASE_DIR.joinpath('Dataset', 'Train')
TEST_DIR    = BASE_DIR.joinpath('Dataset', 'Test')

# File locations
CLASSES_LIST    = BASE_DIR.joinpath('Classes_List.json')

# Constant definitions
IMG_HEIGHT  = 32
IMG_WIDTH   = 32

