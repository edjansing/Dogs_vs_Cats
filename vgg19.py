# -*- coding: utf-8 -*-
"""
FILENAME:   vgg19.py
AUTHOR:     dave
ORG:        OmniEarth, Inc.
DATE:       9/9/16, 07:44
VERSION:    0.01
PY VER:     2.7.X

Info:       VGG19 from K. Simonyan and A. Zisserman, “Very Deep Convolutional Networks for Large-Scale Image
            Recognition,” arXiv.org, vol. cs.CV. 04-Sep-2014.  This architecture is illustrated in Table 1, Column E.

Dependencies:
            keras (using Theano backend)

Versioning:
            0.01:  Initial version
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

def VGG19(binary=False):
    model = Sequential()

    # Layers 1 & 2
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 224, 224)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Layers 3 & 4
    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Layers 5-8
    model.add(Convolution2D(256, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Layers 9-12
    model.add(Convolution2D(512, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Layers 13-16
    model.add(Convolution2D(512, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Layers 17-19 (Fully connected layers)
    model.add(Flatten())     # Note: Keras does automatic shape inference.
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Let user pick between a binary decision (dog v cat, car v truck) or a set of categories
    if binary:
        model.add(Dense(1))
    else:
        model.add(Dense(1000))
    model.add(Activation('softmax'))

    return model