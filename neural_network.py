import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.constraints import MaxNorm

from sklearn.model_selection import KFold, TimeSeriesSplit, LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import shap


def neuralNetwork(x_shape):
    model = Sequential([
        Dense(units=x_shape[1], input_shape=(x_shape[1],), kernel_constraint=MaxNorm(8), kernel_initializer='normal',
              activation='relu'),
        Dense(units=12, kernel_constraint=MaxNorm(8), kernel_initializer='normal', activation='relu'),
        Dropout(0.2),
        Dense(units=1, activation='linear')
    ])

    return model


def neuralNetworkClassify(x_shape):
    model = Sequential([
        Dense(units=x_shape[1], input_shape=(x_shape[1],), kernel_constraint=MaxNorm(5), activation='relu'),
        Dense(units=10, kernel_constraint=MaxNorm(5), activation='relu'),
        Dropout(0.8),
        Dense(units=1, activation='sigmoid')
    ])

    return model


def recurrentNeuralNetwork(segsPerBead,x_shape):
    model = Sequential()
    model.add(LSTM(20, input_shape=[segsPerBead, x_shape[1]], stateful=False, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='linear', kernel_initializer='normal'))

    return model


def plot_NN(Y_true,Y_pred,outputPath):
    plt.rcParams.update({'font.size': 16})
    compPlot = plt.figure()
    line = np.linspace(min(Y_true), max(Y_true), 100)
    plt.plot(line, line)
    plt.scatter(Y_true, Y_pred)
    plt.xlabel('Measured (mm)')
    plt.ylabel('Predicted (mm)')
    compPlot.savefig(outputPath + '\\ComparisonPlot.png')
    plt.show()


def compute_shap(model,X_train,X_test):

    randsamples = X_train[np.random.choice(X_train.shape[0], 150, replace=False)]
    explain = shap.DeepExplainer(model, randsamples)
    shap_vals = explain.shap_values(X_test[0:15])

    return shap_vals[0]


def neuralNetworkMain(X, Y, outputPath, modelType = 'NN', epochs = 700 ,lr = 1e-4, foldSplits = 25, numBeads = None, segsPerBead = None):

    # Get feature names and current number of features
    X_features = X.columns
    number_of_features = len(X_features)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = X.to_numpy()
    X = scaler.fit_transform(X)

    x_shape = X.shape
    cvscores = []
    Y_pred = []
    Y_true = []

    if modelType == 'NN':
        modelInit = neuralNetwork(x_shape)
        kfolds = KFold(n_splits=foldSplits)
    elif modelType == 'NN Classify':
        kfolds = StratifiedKFold(n_splits= foldSplits)
        modelInit = neuralNetworkClassify(x_shape)
    elif modelType == 'LSTM':
        kfolds = KFold(n_splits=foldSplits)
        X = X.reshape(numBeads, segsPerBead, x_shape[1])
        Y = Y.reshape(numBeads, segsPerBead)
        modelInit = recurrentNeuralNetwork(segsPerBead, x_shape)

    shap_vals_folds = np.zeros([15,x_shape[1]])

    for trainIdx, testIdx in kfolds.split(X,Y):
        print('Test Indexes:')
        print(testIdx)

        X_train = X[trainIdx]
        X_test = X[testIdx]

        model = modelInit

        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="mean_squared_error",
            metrics="mse")

        model.fit(X[trainIdx], Y[trainIdx],
                  epochs=epochs,
                  verbose=0,
                  shuffle=True)

        train_scores = model.evaluate(X[trainIdx],Y[trainIdx], verbose = 1)
        scores = model.evaluate(X[testIdx],Y[testIdx], verbose = 1)
        print(scores)
        yFoldPred = model.predict(X[testIdx])

        if modelType == 'LSTM':
            yFoldPred = yFoldPred.reshape(1,segsPerBead*len(testIdx))[0]
            yFoldTrue = Y[testIdx].reshape(1,segsPerBead*len(testIdx))[0]
        else:
            shap_vals = compute_shap(model,X_train,X_test)
            shap_vals_folds = np.add(shap_vals_folds,shap_vals)
            yFoldTrue = Y[testIdx]

        Y_pred.extend(yFoldPred.tolist())
        Y_true.extend(yFoldTrue.tolist())

        cvscores.append(scores[1])
        print('Current split iteration training score: %.6f' % (train_scores[1]))
        print('Current split iteration validation: %.6f' % (scores[1]))
        print('Model overall validation score: %.6f' % (np.mean(cvscores)))


    finalScore = np.mean(cvscores)

    if modelType == 'NN':
        #Compute average shap values across all folds
        shap_vals_avg = shap_vals_folds / foldSplits
        shap_avg = np.mean(shap_vals_avg, axis=0)
        sorted_idx = shap_avg.argsort()
        plt.figure()
        shap.summary_plot(shap_vals_avg, features=X_train, feature_names=X_features, plot_type="bar", max_display=30)
        plt.savefig(outputPath + '\\Shap Values.png')
        print(shap_avg)

        least_valuable_feature = X_features[sorted_idx[0]]
        print('The least valuable feature is: %s' % (least_valuable_feature))

    modelSummary = pd.DataFrame()

    modelSummary['Epochs'] = [epochs]
    modelSummary['Learning Rate'] = [lr]
    modelSummary['Cross Val Score'] = [finalScore]
    modelSummary.to_csv(outputPath + '\\ModelSummary.csv',index = 0)
    print(modelSummary)

    print('Final MSE score: %.5f' % (np.mean(cvscores)))
    print(cvscores)

    config = model.get_config()
    print(config)

    plot_NN(Y_true, Y_pred, outputPath)

    return Y_pred





# def recurrentNeuralNetwork(X,Y,numBeads,segsPerBead,outputPath):
#     try:
#         outputPath = os.path.join(outputPath, 'LSTM')
#         os.mkdir(outputPath)
#     except:
#         pass
#
#     tss = KFold(n_splits = 25)
#     x_shape = X.shape
#
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     X = X.to_numpy()
#     X = scaler.fit_transform(X)
#     X = X.to_numpy().reshape(numBeads, segsPerBead, x_shape[1])
#     Y = Y.reshape(numBeads, segsPerBead)
#     cvscores = []
#     Y_true = []
#     Y_pred = []
#
#     for trainIdx, testIdx in tss.split(X,Y):
#         model = Sequential()
#         model.add(LSTM(20,input_shape = [segsPerBead,x_shape[1]], stateful = False, return_sequences = True ))
#         model.add(Dropout(0.2))
#         model.add(Dense(units=1, activation = 'linear', kernel_initializer='normal'))
#
#         model.compile(
#             optimizer=Adam(learning_rate=1e-4),
#             loss="mean_squared_error",
#             metrics="mse")
#
#         model.fit(X[trainIdx], Y[trainIdx],
#                   epochs=1000,
#                   verbose=0,
#                   shuffle = False)
#
#         scores = model.evaluate(X[testIdx], Y[testIdx], verbose=0)
#         yFoldPred = model.predict(X[testIdx])
#         #yFoldPred = Y_pred.reshape(1, yFoldPred.size)
#         Y_pred.extend(yFoldPred.tolist())
#         Y_true.extend(Y[testIdx])
#         cvscores.append(scores[1])
#
#         train_scores = model.evaluate(X[trainIdx], Y[trainIdx], verbose=0)
#
#         print('Current split iteration training score: %.6f' % (train_scores[1]))
#         print('Current split iteration validation: %.6f' % (scores[1]))
#         print('Model overall validation score: %.6f' % (np.mean(cvscores)))
#
#     print('Final MSE score: %.5f' % (np.mean(cvscores)))
#
#     config = model.get_config()
#     print(config)
#
#
#     Y_true = Y.reshape(1, Y.size)
#     line = np.linspace(min(Y), max(Y), 100)
#     plt.figure()
#     plt.plot(line, line)
#     plt.scatter(Y_true, Y_pred)
#     plt.xlabel('Measured (um)')
#     plt.ylabel('Predicted (um)')
#     plt.show()
#
#     return Y_pred
#
#
# def neuralNetwork(X,Y,outputPath):
#
#     try:
#         outputPath = os.path.join(outputPath, 'NN')
#         os.mkdir(outputPath)
#     except:
#         pass
#
#     kfolds = KFold(n_splits = 25)
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     X = X.to_numpy()
#     X = scaler.fit_transform(X)
#
#     x_shape = X.shape
#     cvscores = []
#     Y_pred = []
#     Y_true = []
#
#     for trainIdx, testIdx in kfolds.split(X,Y):
#         print('Test Indexes:')
#         print(testIdx)
#
#         # X_train = X[trainIdx]
#         # X_test = X[testIdx]
#         # Y_train = Y[trainIdx]
#         # Y_test = Y[testIdx]
#         model = Sequential([
#             Dense(units=x_shape[1], input_shape=(x_shape[1],),kernel_constraint = MaxNorm(8), kernel_initializer='normal', activation='relu'),
#             Dense(units=12, kernel_constraint = MaxNorm(8),kernel_initializer='normal',activation='relu'),
#             Dropout(0.2),
#             Dense(units=1, activation = 'linear')
#         ])
#
#         model.compile(
#             optimizer=Adam(learning_rate=1e-4),
#             loss="mean_squared_error",
#             metrics="mse")
#
#         model.fit(X[trainIdx], Y[trainIdx],
#                   epochs=700,
#                   verbose=0,
#                   shuffle=True)
#         train_scores = model.evaluate(X[trainIdx],Y[trainIdx], verbose = 1)
#         scores = model.evaluate(X[testIdx],Y[testIdx], verbose = 1)
#         yFoldPred = model.predict(X[testIdx])
#         Y_pred.extend(yFoldPred.tolist())
#         Y_true.extend(Y[testIdx])
#         cvscores.append(scores[1])
#         print('Current split iteration training score: %.6f' % (train_scores[1]))
#         print('Current split iteration validation: %.6f' % (scores[1]))
#         print('Model overall validation score: %.6f' % (np.mean(cvscores)))
#
#         # letters = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
#         # explain = shap.DeepExplainer(model, letters)
#         # our_values_for_shap = explain.shap_values(X_test[1:5])
#         # shap.image_plot(our_values_for_shap, -X_test[1:5],show = False)
#
#     print('Final MSE score: %.5f' % (np.mean(cvscores)))
#     print(cvscores)
#
#     config = model.get_config()
#     print(config)
#
#     compPlot = plt.figure()
#     line = np.linspace(min(Y), max(Y), 100)
#     plt.plot(line, line)
#     plt.scatter(Y_true,Y_pred)
#     plt.xlabel('Measured (um)')
#     plt.ylabel('Predicted (um)')
#     compPlot.savefig(outputPath + '\\ComparisonPlot.png')
#
#     plt.show()
#
#     return Y_pred
#
#
# def neuralNetwork_classify(X,Y,outputPath):
#
#     try:
#         outputPath = os.path.join(outputPath, 'NN Classify')
#         os.mkdir(outputPath)
#     except:
#         pass
#
#     #X = X.iloc[:, np.random.choice(X.shape[1], 10, replace=False)]
#
#     #Define train/test splits
#     kfolds = StratifiedKFold(n_splits = 15)
#
#     #Get feature names and current number of features
#     X_features = X.columns
#     number_of_features = len(X_features)
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     X = X.to_numpy()
#     X = scaler.fit_transform(X)
#
#
#     x_shape = X.shape
#     cvscores = []
#     Y_pred = []
#     Y_true = []
#
#
#
#     for trainIdx, testIdx in kfolds.split(X,Y):
#         print('Test Indexes:')
#         print(testIdx)
#
#         X_train = X[trainIdx]
#         X_test = X[testIdx]
#         Y_train = Y[trainIdx]
#         Y_test = Y[testIdx]
#         model = Sequential([
#             Dense(units=x_shape[1], input_shape=(x_shape[1],), kernel_constraint=MaxNorm(5), activation='relu'),
#             Dense(units=10, kernel_constraint=MaxNorm(5), activation='relu'),
#             Dropout(0.8),
#             Dense(units=1, activation = 'sigmoid')
#         ])
#
#         model.compile(
#             optimizer=Adam(learning_rate=1e-3),
#             loss="binary_crossentropy",
#             metrics="accuracy")
#
#         model.fit(X[trainIdx], Y[trainIdx],
#                   epochs=50,
#                   verbose=0,
#                   shuffle=True)
#
#         train_scores = model.evaluate(X[trainIdx],Y[trainIdx], verbose = 1)
#         scores = model.evaluate(X[testIdx],Y[testIdx], verbose = 1)
#         yFoldPred = model.predict(X[testIdx])
#         Y_pred.extend(yFoldPred.tolist())
#         Y_true.extend(Y[testIdx])
#         cvscores.append(scores[1])
#         print('Current split iteration training score: %.6f' % (train_scores[1]))
#         print('Current split iteration validation: %.6f' % (scores[1]))
#         print('Model overall validation score: %.6f' % (np.mean(cvscores)))
#
#         randsamples = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
#         explain = shap.DeepExplainer(model,randsamples)
#         shap_vals = explain.shap_values(X_test[1:5])
#         shap_avg = np.mean(shap_vals[0], axis = 0)
#         sorted_idx = shap_avg.argsort()
#
#         least_valuable_feature = X_features[sorted_idx[0]]
#         print('The least valuable feature is: %s' % (least_valuable_feature))
#         #X_new = X_new.drop(least_valuable_feature, axis=1)
#         plt.figure()
#         shap.summary_plot(shap_vals, features=X_train, feature_names= X_features, plot_type="bar", max_display=30)
#         plt.savefig(outputPath + '\\Shap Values.png')
#
#     config = model.get_config()
#     print(config)
#
#     print('Final MSE score: %.5f' % (np.mean(cvscores)))
#     print(cvscores)
#
#     # saveScores = pd.DataFrame(columns = 'KFold Scores')
#     # saveScores
#
#     return Y_pred
















# model.add(SimpleRNN(
#             units=x_shape[1],
#             activation="tanh",
#             kernel_constraint = MaxNorm(3),
#             use_bias=True,
#             return_sequences=True))
#model.add(Dense(units=12, kernel_constraint=MaxNorm(3), activation='relu'))