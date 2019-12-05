#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 3 15:45:43 2019
@author: georgebarker and andrezeromski
"""

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, SimpleRNN, Dropout, Conv1D, MaxPooling1D
import numpy as np

# constants
top_words = 5000
max_review_length = 600
embedding_vector_length = 64

## Load Data
## save np.load
#np_load_old = np.load
#
## modify the default parameters of np.load
#np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
#
## call load_data with allow_pickle implicitly set to true
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=8000)
#
## restore np.load for future normal usage
#np.load = np_load_old

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)

# Pad and reduce length of input
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

# Create model
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(filters=1, kernel_size=3, strides=1, activation='relu', padding='valid'))
model.add(MaxPooling1D(pool_size=2))
model.add(SimpleRNN(64))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Summary of model
print(model.summary())

# Train model
model.fit(x_train, y_train, epochs=4, batch_size=64, verbose=1)

# Evaluate model
predictions = model.evaluate(x_test, y_test)
print("accuracy: %.2f%%" % (predictions[1]*100))
