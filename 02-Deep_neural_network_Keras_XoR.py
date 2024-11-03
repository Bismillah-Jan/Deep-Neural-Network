#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 20:16:40 2017

@author: bismillah
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])


model = Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid', init='glorot_uniform'))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X, y, show_accuracy=True, batch_size=1, nb_epoch=1000)
print model.predict(X).round()
