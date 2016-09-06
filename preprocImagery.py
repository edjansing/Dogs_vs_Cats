# -*- coding: utf-8 -*-
"""
FILENAME:   preprocImagery.py
AUTHOR:     dave
ORG:        OmniEarth, Inc.
DATE:       9/5/16, 08:11
VERSION:    0.01
PY VER:     2.7.X

Info:       Preprocess the dogs vs cats imagery into training/test bundles

Dependencies:
            skimage
            numpy
            glob
            os.path
            tqdm (a really cool progress bar)

Versioning:
            0.01:  Initial version
"""

import numpy as np
from skimage.io import imread
from skimage.transform import resize
import glob
from os.path import join
from tqdm import tqdm
from random import shuffle
import time

def centerCrop(img):
    # Borrowed from "VGG in TensorFlow · Davi Frossard"
    image_h, image_w, _ = np.shape(img)
    shorter_side = min(image_h, image_w)
    scale = 224. / shorter_side
    image_h, image_w = np.ceil([scale * image_h, scale * image_w]).astype('int32')
    img = resize(img, (image_h, image_w))
    crop_x = (image_w - 224) / 2
    crop_y = (image_h - 224) / 2
    img = img[crop_y:crop_y + 224,crop_x:crop_x + 224, :]

    return img

def returnFileList(inDir, pattName):
    return glob.glob(join(inDir, pattName))

if __name__ == '__main__':
    # dataDir = '/Users/dave/Data/DogsvCats/train'
    dataDir = '/data/DogsvCats/train'

    dogFileList = returnFileList(dataDir, 'dog*')
    catFileList = returnFileList(dataDir, 'cat*')
    numberSamples = len(dogFileList) + len(catFileList)
    print('Total number of samples = %d' % numberSamples)

    # How much to hold out?
    testSampleSize = int(np.ceil(numberSamples * 0.10))
    print('Total number to hold out for test = %d' % testSampleSize)

    # Allocate numpy arrays
    images = np.zeros((numberSamples, 224, 224, 3), dtype='float32')
    labels = np.zeros((numberSamples), dtype='float32')

    # Get imagery, center crop, store in array
    for k in tqdm(range(len(dogFileList))):
        img = imread(dogFileList[k])
        images[k, :, :, :] = centerCrop(img)
        labels[k] = +1.0
    for k in tqdm(range(len(catFileList))):
        img = imread(catFileList[k])
        images[k + len(dogFileList), :, :, :] = centerCrop(img)
        labels[k + len(dogFileList)] = 0.0

    k = range(numberSamples)
    shuffle(k)
    images = images[k, :, :, :]
    labels = labels[k]

    testImages = images[:testSampleSize, :, :, :]
    testLabels = labels[:testSampleSize]
    trainImages = images[testSampleSize:, :, :, :]
    trainLabels = labels[testSampleSize:]

    # Save off results
    np.savez_compressed('/data/DogsvCats/DvC_test.npz', testImages=testImages, testLabels=testLabels)
    for k in range(0, numberSamples - testSampleSize, 2500):
        tmpImages = trainImages[k:k+2500, :, :, :]
        tmpLabels = trainLabels[k:k+2500]
        tmpStr = '/data/DogsvCats/DvC_train_%d_to_%d.npz' % (k, k + 2500 - 1) 
    	np.savez_compressed(tmpStr, trainImages=tmpImages, trainLabels=tmpLabels)

