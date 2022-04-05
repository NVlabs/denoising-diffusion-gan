# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from inception score
#
# Source:
# https://github.com/tsc2017/Inception-Score/blob/04390da2ebb3c9a3860337a33297c4f270bd906d/inception_score.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_inception).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------


'''
Usage:
    Call get_inception_score(images, splits=10)
Args:
    images: A numpy array with values ranging from 0 to 255 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary. A dtype of np.uint8 is recommended to save CPU memory.
    splits: The number of splits of the images, default is 10.
Returns:
    Mean and standard deviation of the Inception Score across the splits.
'''
import argparse

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_gan as tfgan
import os
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
# pip install tensorflow-gan
import tensorflow_gan as tfgan
session=tf.compat.v1.InteractiveSession()
# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 64
INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'

# Run images through Inception.
inception_images = tf.compat.v1.placeholder(tf.float32, [None, 3, None, None], name = 'inception_images')
def inception_logits(images = inception_images, num_splits = 1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.compat.v1.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
    logits = tf.map_fn(
        fn = tfgan.eval.classifier_fn_from_tfhub(INCEPTION_TFHUB, INCEPTION_OUTPUT, True),
        elems = array_ops.stack(generated_images_list),
        parallel_iterations = 8,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits

logits=inception_logits()

def get_inception_probs(inps):
    session=tf.get_default_session()
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    preds = np.zeros([inps.shape[0], 1000], dtype = np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] / 255. * 2 - 1
        preds[i * BATCH_SIZE : i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = session.run(logits,{inception_images: inp})[:, :1000]
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    return preds

def preds2score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def get_inception_score(images, splits=10):
    assert(type(images) == np.ndarray)
    assert(len(images.shape) == 4)
    assert(images.shape[1] == 3)
    assert(np.min(images[0]) >= 0 and np.max(images[0]) > 10), 'Image values should be in the range [0, 255]'
    print('Calculating Inception Score with %i images in %i splits' % (images.shape[0], splits))
    start_time=time.time()
    preds = get_inception_probs(images)
    mean, std = preds2score(preds, splits)
    print('Inception Score calculation time: %f s' % (time.time() - start_time))
    return mean, std  # Reference values: 11.38 for 50000 CIFAR-10 training set images, or mean=11.31, std=0.10 if in 10 splits.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_dir', default='./saved_samples/', help='path to saved images')
    opt = parser.parse_args()

    data = np.load(opt.sample_dir)
    data = np.clip(data, 0, 255)
    m, s = get_inception_score(data, splits=1)
    
    print('mean: ', m)
    print('std: ', s)
