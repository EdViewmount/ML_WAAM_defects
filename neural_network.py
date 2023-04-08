import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

import data_processing as dp

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, SimpleRNN, Dropout
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.constraints import MaxNorm

from sklearn.model_selection import KFold, TimeSeriesSplit, LeaveOneOut, StratifiedKFold

import shap


def recurrentNeuralNetwork(X,Y,numBeads,segsPerBead):

    tss = TimeSeriesSplit(n_splits = 10)
    x_shape = X.shape
    X = dp.normalize_data(X)
    X = X.to_numpy().reshape(numBeads, segsPerBead, x_shape[1])
    Y = Y.reshape(numBeads, segsPerBead)
    cvscores = []
    Y_true = []
    Y_pred = []

    for trainIdx, testIdx in tss.split(X,Y):
        model = Sequential()
        model.add(SimpleRNN(
            units=x_shape[1],
            activation="tanh",
            kernel_constraint = MaxNorm(3),
            use_bias=True,
            return_sequences=True))
        Dense(units=15, kernel_constraint=MaxNorm(3), activation='relu'),
        Dropout(0.2),
        model.add(Dense(units=1, activation = 'linear', kernel_initializer='normal'))

        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="mean_squared_error",
            metrics="mse")

        model.fit(X[trainIdx], Y[trainIdx],
                  epochs=100,
                  verbose="auto")

        scores = model.evaluate(X[testIdx], Y[testIdx], verbose="auto")
        yFoldPred = model.predict(X[testIdx])
        #yFoldPred = Y_pred.reshape(1, yFoldPred.size)
        Y_pred.extend(yFoldPred.tolist())
        Y_true.extend(Y[testIdx])
        cvscores.append(scores[1])

        train_scores = model.evaluate(X[trainIdx], Y[trainIdx], verbose=1)

        print('Current split iteration training score: %.6f' % (train_scores[1]))
        print('Current split iteration validation: %.6f' % (scores[1]))
        print('Model overall validation score: %.6f' % (np.mean(cvscores)))

    print('Final MSE score: %.5f' % (np.mean(cvscores)))


    Y_true = Y.reshape(1, Y.size)
    line = np.linspace(min(Y), max(Y), 100)
    plt.figure()
    plt.plot(line, line)
    plt.scatter(Y_true, Y_pred)
    plt.xlabel('Measured (um)')
    plt.ylabel('Predicted (um)')
    plt.show()


def neuralNetwork(X,Y):

    kfolds = KFold(n_splits = 10)
    X = dp.normalize_data(X)
    X = X.to_numpy()

    x_shape = X.shape
    cvscores = []
    Y_pred = []
    Y_true = []

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1)

    for trainIdx, testIdx in kfolds.split(X,Y):
        print('Test Indexes:')
        print(testIdx)

        # X_train = X[trainIdx]
        # X_test = X[testIdx]
        # Y_train = Y[trainIdx]
        # Y_test = Y[testIdx]
        model = Sequential([
            Dense(units=x_shape[1], input_shape=(x_shape[1],), kernel_constraint=MaxNorm(6), kernel_initializer='normal', activation='relu'),
            Dense(units=13, kernel_constraint=MaxNorm(6), kernel_initializer='normal',activation='relu'),
            Dropout(0.2),
            Dense(units=1, activation = 'linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="mean_squared_error",
            metrics="mse")

        model.fit(X[trainIdx], Y[trainIdx],
                  epochs=200,
                  verbose=0,
                  shuffle=True)
        train_scores = model.evaluate(X[trainIdx],Y[trainIdx], verbose = 1)
        scores = model.evaluate(X[testIdx],Y[testIdx], verbose = 1)
        yFoldPred = model.predict(X[testIdx])
        Y_pred.extend(yFoldPred.tolist())
        Y_true.extend(Y[testIdx])
        cvscores.append(scores[1])
        print('Current split iteration training score: %.6f' % (train_scores[1]))
        print('Current split iteration validation: %.6f' % (scores[1]))
        print('Model overall validation score: %.6f' % (np.mean(cvscores)))

        # letters = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        # explain = shap.DeepExplainer(model, letters)
        # our_values_for_shap = explain.shap_values(X_test[1:5])
        # shap.image_plot(our_values_for_shap, -X_test[1:5])

    print('Final MSE score: %.5f' % (np.mean(cvscores)))
    print(cvscores)

    line = np.linspace(min(Y), max(Y), 100)
    plt.figure()
    plt.plot(line, line)
    plt.scatter(Y_true,Y_pred)
    plt.xlabel('Measured (um)')
    plt.ylabel('Predicted (um)')
    plt.show()


def neuralNetwork_classify(X,Y):

    #X = X.iloc[:, np.random.choice(X.shape[1], 10, replace=False)]

    #Define train/test splits
    kfolds = StratifiedKFold(n_splits = 10)

    #Get feature names and current number of features
    X_features = X.columns
    number_of_features = len(X_features)
    X = dp.normalize_data(X)
    X = X.to_numpy()


    x_shape = X.shape
    cvscores = []
    Y_pred = []
    Y_true = []



    for trainIdx, testIdx in kfolds.split(X,Y):
        print('Test Indexes:')
        print(testIdx)

        X_train = X[trainIdx]
        X_test = X[testIdx]
        Y_train = Y[trainIdx]
        Y_test = Y[testIdx]
        model = Sequential([
            Dense(units=x_shape[1], input_shape=(x_shape[1],), kernel_constraint=MaxNorm(5), activation='relu'),
            Dense(units=10, kernel_constraint=MaxNorm(5), activation='relu'),
            Dropout(0.8),
            Dense(units=1, activation = 'sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics="accuracy")

        model.fit(X[trainIdx], Y[trainIdx],
                  epochs=25,
                  verbose=0,
                  shuffle=True)

        train_scores = model.evaluate(X[trainIdx],Y[trainIdx], verbose = 1)
        scores = model.evaluate(X[testIdx],Y[testIdx], verbose = 1)
        yFoldPred = model.predict(X[testIdx])
        Y_pred.extend(yFoldPred.tolist())
        Y_true.extend(Y[testIdx])
        cvscores.append(scores[1])
        print('Current split iteration training score: %.6f' % (train_scores[1]))
        print('Current split iteration validation: %.6f' % (scores[1]))
        print('Model overall validation score: %.6f' % (np.mean(cvscores)))

        randsamples = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        explain = shap.DeepExplainer(model,randsamples)
        shap_vals = explain.shap_values(X_test[1:5])
        shap_avg = np.mean(shap_vals[0], axis = 0)
        sorted_idx = shap_avg.argsort()

        least_valuable_feature = X_features[sorted_idx[0]]
        print('The least valuable feature is: %s' % (least_valuable_feature))
        #X_new = X_new.drop(least_valuable_feature, axis=1)
        #plt.figure()
        #shap.summary_plot(shap_vals, features=X_train, feature_names= X_features, plot_type="bar", max_display=30)

    print('Final MSE score: %.5f' % (np.mean(cvscores)))
    print(cvscores)

