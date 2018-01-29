# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 05:51:39 2018

@author: Cody Hart
@about: Tests the digits_model for accuracy
"""

import numpy as np
np.random.seed(123)

from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import load_model


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


X_test = X_test.astype('float32')

X_test /= 255

y_test = np_utils.to_categorical(y_test, 10)


model = load_model('digits_model.h5')
score = model.evaluate(X_test, y_test,verbose=0)

print(score)