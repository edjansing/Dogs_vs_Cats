# -*- coding: utf-8 -*-
"""
FILENAME:   trainWithVGG19forDvC.py
AUTHOR:     dave
ORG:        OmniEarth, Inc.
DATE:       9/9/16, 08:00
VERSION:    0.01
PY VER:     2.7.X

Info:       Train two categories:  dog v cat with Keras VGG19 model.

Dependencies:
            VGG19
            numpy
            keras (Using Theano backend)
            os.path

Versioning:
            0.01:  Initial version
"""

import vgg19
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from os.path import join

#-----------------------------------------------------------------------------------------------------------------------
# Get the data
#-----------------------------------------------------------------------------------------------------------------------
print('Get the data . . . ')

dataDir = '/data/DogsvCats'

# Load up training data
trainRoot = 'DvC_train_*'
testFile = 'DvC_test.npz'
trainingFileList = glob.glob(join(dataDir, trainRoot))

trainImages = []
trainLabels = []

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

# Keras expects the imagery to be in (channel x row x col) format
trainImages = np.transpose(trainImages, (0, 3, 1, 2))
testImages = np.transpose(testImages, (0, 3, 1, 2))

print('Done!')

#-----------------------------------------------------------------------------------------------------------------------
# Make the model
#-----------------------------------------------------------------------------------------------------------------------
print('Set up model . . . ')
model = vgg19.VGG19(binary=True)
print('Done!')

#-----------------------------------------------------------------------------------------------------------------------
# Set up the SGD optimizer and compile model
#-----------------------------------------------------------------------------------------------------------------------
print('Set up SGD optimizer and compile . . .')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)
print('Done!')

#-----------------------------------------------------------------------------------------------------------------------
# Train
#-----------------------------------------------------------------------------------------------------------------------
print('Train . . .')
model.fit(trainImages, trainLabels, validation_split=0.10, verbose=1, batch_size=256, nb_epoch=1)
print('Done!')

#-----------------------------------------------------------------------------------------------------------------------
# Save
#-----------------------------------------------------------------------------------------------------------------------
print('Save weights . . . ')
model.save_weights('/Users/dave/Data/DogsvCats/20160908_kerasVGG19_dvc')
print('Done!')
