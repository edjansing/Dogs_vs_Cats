# -*- coding: utf-8 -*-
"""
FILENAME:   DvCClassifier.py
AUTHOR:     dave
ORG:        OmniEarth, Inc.
DATE:       9/6/16, 06:56
VERSION:    0.01
PY VER:     2.7.X

Info:       This file defines the classifier that will be used for Dogs v Cat classification.

Dependencies:
            numpy
            tensorflow

Versioning:
            0.01:  Initial version
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

# maxPool2x2:  Performs a 2 x 2 max pooling on a tensor
def max_pool_2x2(inTensor):
    return tf.nn.max_pool(
        inTensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# This defines the Convolutional Neural Network (CNN).  Based on example from Tensorflow/skflow
# Github page.
def conv_model(X, y):
    # reshape X to 4d tensor with 2nd and 3rd dimensions being image width and
    # height final dimension being the number of color channels.
    X = tf.reshape(X, [-1, 224, 224, 3])
    with tf.variable_scope('pre-process'):
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        X = X - mean

    with tf.variable_scope('conv_layer1'):
        h_conv1 = learn.ops.conv2d(X, n_filters=32, filter_shape=[5, 5],
                               bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope('conv_layer2'):
        h_conv2 = learn.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5],
                                   bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # densely connected layer with 1024 neurons.
    h_fc1 = learn.ops.dnn(
        h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)

    return learn.models.logistic_regression(h_fc1, y)
