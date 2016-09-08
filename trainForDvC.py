# -*- coding: utf-8 -*-
"""
FILENAME:   trainForDvC.py
AUTHOR:     dave
ORG:        OmniEarth, Inc.
DATE:       9/6/16, 14:22
VERSION:    0.01
PY VER:     2.7.X

Info:       Script to pull in training/test data, train for Dogs vs Cats

Dependencies:
            Python Package #1
            Python Package #2
            Python Package #3

Versioning:
            0.01:  Initial version
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import DvCClassifier as dc
import glob
from os.path import join
from tqdm import tqdm

dataDir = '/data/DogsvCats'

# Load up training data
trainRoot = 'DvC_train_*'
testFile = 'DvC_test.npz'
trainingFileList = glob.glob(join(dataDir, trainRoot))

trainImages = []
trainLabels = []
print('Extracting training/test data . . .')
for k in tqdm(range(len(trainingFileList))):
    data = np.load(trainingFileList[k])
    if not k:
        trainImages = data['trainImages']
        trainLabels = data['trainLabels']
    else:
        trainImages = np.append(trainImages, data['trainImages'], axis = 0)
        trainLabels = np.append(trainLabels, data['trainLabels'], axis = 0)

data = np.load(join(dataDir, testFile))
testImages = data['testImages']
testLabels = data['testLabels']

del data

classifier = learn.TensorFlowEstimator(
    model_fn=dc.conv_model, n_classes=2, batch_size=100, steps=5000,
    learning_rate=0.0001, verbose=1)
classifier.fit(trainImages, trainLabels, logdir='/data/DogsvCats/20160906_DvC_CNN/')
# classifier.fit(trainImages, trainLabels)
# classifier.save('20160906_DvC_CNN')
