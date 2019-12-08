#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 3 15:45:43 2019
@author: George Barker and Andre Zeromski
"""
import pickle
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, SimpleRNN, Dropout, Conv1D, MaxPooling1D, Bidirectional, LSTM, BatchNormalization
from keras.regularizers import l2

# constants
top_words = 4000
max_review_length = 800
embedding_vector_length = 50

## Load Data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)

# Pad and reduce length of input
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

# Create model
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.3))
##model.add(Dense(128, activation="relu"))
##model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Summary of model
print(model.summary())

# Train model
model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=8, batch_size=32, verbose=1)

# Evaluate model
predictions = model.evaluate(x_test, y_test)
print("accuracy: %.2f%%" % (predictions[1]*100))

pickle_out = open("CNN-LSTM-1","wb")
pickle.dump(model, pickle_out)
pickle_out.close()
