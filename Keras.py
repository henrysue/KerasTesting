# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:27:28 2020

@author: HSAdmin
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.dataset.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

