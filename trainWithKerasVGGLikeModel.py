# -*- coding: utf-8 -*-
"""
FILENAME:   trainWithKerasVGGLikeModel.py
AUTHOR:     dave
ORG:        OmniEarth, Inc.
DATE:       9/8/16, 20:39
VERSION:    0.01
PY VER:     2.7.X

Info:       This uses Keras (with Theano backend) to build a VGG-like network and train on the Dawgs v Kats
            data set.  Borrowed heavily from keras.io.

Dependencies:
            numpy
            keras (using Theano backend)
            os.path

Versioning:
            0.01:  Initial version
"""

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
dataDir = '/Users/dave/Data/DogsvCats'
# testFile = 'DvC_test.npz'
#
# data = np.load(join(dataDir, testFile))
# testImages = data['testImages']
# testLabels = data['testLabels']

trainFile = 'DvC_train_0_to_2499.npz'
data = np.load(join(dataDir, trainFile))
trainImages = data['trainImages']
trainLabels = data['trainLabels']

# Keras expects the imagery to be in (channel x row x col) format
# testImages = np.transpose((2, 0, 1))
trainImages = np.transpose(trainImages, (0, 3, 1, 2))
print('Done!')

#-----------------------------------------------------------------------------------------------------------------------
# Make the model
#-----------------------------------------------------------------------------------------------------------------------
print('Set up model . . . ')
model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 224, 224)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('softmax'))
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
model.fit(trainImages, trainLabels, validation_split=0.25, verbose=1, batch_size=32, nb_epoch=1)
print('Done!')

#-----------------------------------------------------------------------------------------------------------------------
# Save
#-----------------------------------------------------------------------------------------------------------------------
print('Save weights . . . ')
model.save_weights('/Users/dave/Data/DogsvCats/20160908_kerasVGGlikemodel_dvc')
print('Done!')
