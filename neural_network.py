import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

import data_processing as dp

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, SimpleRNN
from tensorflow.python.keras.optimizer_v2.adam import Adam

from sklearn.model_selection import KFold

def recurrentNeuralNetwork(X,Y,numBeads,segsPerBead):
    x_shape = X.shape
    X = dp.normalize_data(X)
    X = X.to_numpy().reshape(numBeads,segsPerBead,x_shape[1])
    Y = np.array(Y).reshape(numBeads,segsPerBead)

    model = Sequential()
    model.add(SimpleRNN(
        units = x_shape[1],
        activation="tanh",
        use_bias=True,
        return_sequences=True))
    model.add(Dense(units=10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='normal'))

    model.compile(
        optimizer=Adam(learning_rate=0.0001, decay = 1e-8),
        loss="mean_squared_error",
        metrics="mse")

    model.fit(X, Y,
              epochs=60000,
              verbose="auto",
              shuffle=False)

    y_pred = model.predict(X, workers=1)
    y_pred = y_pred.reshape(1,y_pred.size)
    Y = Y.reshape(1, Y.size)
    line = np.linspace(min(Y), max(Y), 100)
    plt.figure()
    plt.plot(line, line)
    plt.scatter(Y, y_pred)
    plt.xlabel('Measured (um)')
    plt.ylabel('Predicted (um)')
    plt.show()


def neuralNetwork(X,Y):
    X = dp.normalize_data(X)
    X = X.to_numpy()
    Y = np.array(Y)

    x_shape = X.shape

    model = Sequential([
        Dense(units = x_shape[1], input_shape=(x_shape[1],), kernel_initializer = 'normal',  activation = 'relu'),
        Dense(units=10, kernel_initializer='normal', activation='relu'),
        Dense(units = 1, kernel_initializer = 'normal')
    ])

    model.compile(
        optimizer= Adam(learning_rate = 1e-4),
        loss="mean_squared_error",
        metrics="mse")

    model.fit(X,Y,
        batch_size = 32,
        epochs=60000,
        verbose="auto",
        shuffle=False)

    y_pred = model.predict(X,workers=1)
    line = np.linspace(min(Y), max(Y), 100)
    plt.figure()
    plt.plot(line, line)
    plt.scatter(Y,y_pred)
    plt.xlabel('Measured (um)')
    plt.ylabel('Predicted (um)')
    plt.show()