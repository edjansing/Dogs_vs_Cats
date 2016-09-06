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
import DvCClassifier as dc
import glob
from os.path import join

dataDir = '/data/DogsvCats'

# Load up training data
trainRoot = 'DvC_train_*'
testFile = 'DvC_test.npz'
trainingFileList = glob.glob(join(dataDir, trainRoot))

trainImages = []
trainLabels = []
for k in range(len(trainingFileList)):
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
