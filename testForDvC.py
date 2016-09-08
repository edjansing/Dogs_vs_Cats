# -*- coding: utf-8 -*-
"""
FILENAME:   testForDvC.py
AUTHOR:     dave
ORG:        OmniEarth, Inc.
DATE:       9/8/16, 09:54
VERSION:    0.01
PY VER:     2.7.X

Info:       Test the classifier on DvC data

Dependencies:
            Python Package #1
            Python Package #2
            Python Package #3

Versioning:
            0.01:  Initial version
"""

import numpy as np
from tensorflow.contrib import learn
import DvCClassifier as dc
from os.path import join
from sklearn import metrics

dataDir = '/data/DogsvCats'
testFile = 'DvC_test.npz'

data = np.load(join(dataDir, testFile))
testImages = data['testImages']
testLabels = data['testLabels']

classifier = learn.Estimator(model_fn=dc.conv_model, model_dir='/data/DogsvCats/20160907_DvC_CNN_model')
score = metrics.accuracy_score(testLabels, classifier.predict(testImages))