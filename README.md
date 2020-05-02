# Neural Network Implementations
Hand-written characters using multilayer perceptron networks (MLPs) and convolutional neural networks (CNNs).

## Requirements
All of the requirements are handled through Conda, including the correct Python version.
Note that to accelerate training and inferencing through GPU, installation of CUDA 10.1 is required.
See [TensorFlow on GPU support](https://www.tensorflow.org/install/gpu).
GPU installation is **not** required to run this software.
To install the Conda environment:
```
conda env create -f environment.yml
```

Activate the Conda environment:
```
conda activate DIP_Final_Project
```

## Usage
Usage of this software consists of two stages:
- Training the models
- Testing the models

**Important Note:** Running all of these takes a very long time.
For your convenience, we have provided all of our trained models in the Models directory.
Additionally, we have included our network evaluation results in the Training_Logs directory.

### Training
Training the MLP networks is handled in `Model_Trainer.py`.
To run the MLP trainer:
```
python Model_Trainer.py
```
Note that this runs in Python 3, but Conda redefines the `python` environmental variable to point to a local instance.
This may be different on Windows, though (we developed this on Linux).

Similarly, training the CNNs:
```
python Model_Trainer_CNN.py
```

**Note:** Training will take many, many hours for all of the networks (Probably 12+ hours total).
We have provided pre-trained models in the Models directory.

### Testing
Testing the models is handled in `Model_Eval.py`.
The script tests every model in the Models directory, so to test only one model at a time, you can move the rest of the models to a separate directory.
The script outputs confusion matrices and a table summarizing the testing and training accuracy of each network.
To run evaluation:
```
python Model_Eval.py
```

**Note:** While this takes less time than training, it took a couple of hours to evaluate every single model (on CPU, not GPU).
This is because it evaluates the entire training set and testing set, then re-evaluates each image individually for the confusion matrix.
We have provided the calculated metrics for your convenience in the Training_Logs directory.
