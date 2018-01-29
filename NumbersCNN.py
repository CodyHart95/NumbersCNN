# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 03:04:19 2018

@author: Cody Hart
"""

import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

#loads data into two tuples each containing x and y data where
#is the image and y is the image label
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#normalizes our image data into (28x28x1) images
X_train = X_train.reshape(X_train.shape[0],28,28,1)

#converts values to floats and their RGB values into a range 
#between 0 and 1
X_train = X_train.astype('float32')
X_train /= 255

#converts the 1D class arrays into a 10D class matrices
#number classes
y_train = np_utils.to_categorical(y_train, 10)


model = Sequential()

#I honestly have no idea whats going on in this entire block of add statements,
#but it basically creates the structure of our network
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(X_train, y_train, 
          batch_size=32, epochs=10, verbose=1)

model.save('digits_model.h5')