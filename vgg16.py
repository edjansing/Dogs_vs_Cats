# -*- coding: utf-8 -*-
"""
FILENAME:   vgg16.py
AUTHOR:     dave
ORG:        OmniEarth, Inc.
DATE:       9/10/16, 12:30
VERSION:    0.01
PY VER:     2.7.X

Info:       Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam et 
            nisl turpis. Aliquam fringilla, elit vel ornare ultrices, urna 
            tortor consequat tortor, vitae tempus leo velit sed lorem. Lorem 
            ipsum dolor sit amet, consectetur adipiscing elit. Nulla facilisis 
            pharetra lorem, quis bibendum turpis. Duis et arcu dolor...

Dependencies:
            Python Package #1
            Python Package #2
            Python Package #3

Versioning:
            0.01:  Initial version
"""
import os
import h5py
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

def vgg16(backendWeightsFile=None, frontendWeightsFile=None, img_width=150, img_height=150):
    if backendWeightsFile:
        backendTrainableFlag = False
    else:
        backendTrainableFlag = True

    if frontendWeightsFile:
        frontendTrainableFlag = False
    else:
        frontendTrainableFlag = True

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', trainable=backendTrainableFlag))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2', trainable=backendTrainableFlag))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1', trainable=backendTrainableFlag))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2', trainable=backendTrainableFlag))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1', trainable=backendTrainableFlag))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', trainable=backendTrainableFlag))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3', trainable=backendTrainableFlag))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1', trainable=backendTrainableFlag))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2', trainable=backendTrainableFlag))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3', trainable=backendTrainableFlag))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1', trainable=backendTrainableFlag))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2', trainable=backendTrainableFlag))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3', trainable=backendTrainableFlag))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    if backendWeightsFile:
        assert os.path.exists(backendWeightsFile), 'Backend Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(backendWeightsFile)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        print('Backend model loaded.')

    model.add(Flatten())
    model.add(Dense(256, activation='relu', trainable=frontendTrainableFlag, name='Dense1'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', trainable=frontendTrainableFlag, name='Dense2'))

    if frontendWeightsFile:
        assert os.path.exists(frontendWeightsFile), 'Frontend model weights not found'
        f = h5py.File(frontendWeightsFile)
        weights = f['dense_1'].values()
        model.layers[32].set_weights(weights)
        weights = f['dense_2'].values()
        model.layers[34].set_weights(weights)
        f.close()
        print('Frontend model loaded.')

    return model