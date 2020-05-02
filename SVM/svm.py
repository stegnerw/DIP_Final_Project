# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:19:35 2020

@author: Liu
"""

# Import packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pathlib
import pandas

# Import path configs
from Settings import *

# Set number of samples
MAX_TRAIN = 100
MAX_TEST = 100

# Parse class names
train_path = [x for x in TRAIN_DIR.iterdir()]
test_path = [x for x in TEST_DIR.iterdir()]
char_class = [x.name for x in train_path]
class2idx = {u:i for i,u in enumerate(char_class)}
i = 0

print("----------Setup complete----------")

# Import training data
train_img = []
train_true = []
for p in train_path:
    fl = [x for x in p.iterdir()]
    for f in fl:
        
        # i += 1
        # if i > MAX_TRAIN:
        #     i = 0
        #     break
        
        train_img.append(plt.imread(str(f)))
        train_true.append(class2idx[p.name])

print("----------Training data loading complete--------------")

# Import testing data
i = 0
test_img = []
test_true = []
for p in test_path:
    fl = [x for x in p.iterdir()]
    for f in fl:
        
        i += 1
        if i > MAX_TEST:
            i = 0
            break
        
        test_img.append(plt.imread(str(f)))
        test_true.append(class2idx[p.name])
        
print("----------Testing data loading complete--------------")

# Preprocess data to feed into model
train_img = np.array(train_img)
train_true = np.array(train_true)
test_img = np.array(test_img)
test_true = np.array(test_true)
train_data = train_img.reshape((len(train_img), -1))
test_data = test_img.reshape((len(test_img), -1))

# Initialize classifier 
classifier = svm.SVC(kernel='rbf', gamma='auto', verbose=False)
# Train the model by fitting trainning data
classifier.fit(train_data, train_true)
print("----------Learning complete--------------")

# Predict model with testing data
test_pred = classifier.predict(test_data)
print("----------Prediction complete--------------")


# Generate and output result
print("----------Generating report--------------")
CR = classification_report(test_true, test_pred, output_dict=True)
CR_df = pandas.DataFrame.from_dict(CR)
CR_df.to_csv('classification_report.csv')
print("----------Generating Confusion--------------")
CM = confusion_matrix(test_true, test_pred)
CM_df = pandas.DataFrame(CM)
CM_df.to_csv('confusion_matrix.csv')