import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import data_processing as dp

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance


def regression_noRFE(Characterization, X, unitsChar):

    for metric in Characterization:
        Y = Characterization[metric]
        Y_range = np.amax(Y) - np.amin(Y)
        print(Y_range)

        regressor = RandomForestRegressor(n_estimators=1000)
        regressor.fit(X, Y)
        cv = LeaveOneOut()

        n_scores = cross_val_score(regressor, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        predictions = cross_val_predict(regressor, X, Y, cv=cv, n_jobs=-1)

        print(metric)
        print('MAE')
        print(n_scores)
        print('Average:')
        print(n_scores.mean())

        from sklearn.tree import plot_tree

        fig = plt.figure(figsize=(15, 10))
        plot_tree(regressor.estimators_[0],
                  feature_names=X.columns,
                  filled=True, impurity=True,
                  rounded=True)

        fig.savefig('tree' + metric + '.png')

        # Feature Importance
        feat_imp_perm = permutation_importance(regressor, X, Y)

        importances_df = pd.DataFrame(columns=['Feature', 'Importance'])
        DF_features = X.columns.values.tolist()
        importances_df['Feature'] = DF_features
        importances_df['Importance'] = feat_imp_perm.importances_mean
        importances_df.to_csv('Importances_ ' + metric + '.csv')

        line = np.linspace(min(Y), max(Y), 30)
        fig_plot = plt.figure()
        plt.scatter(predictions, Y)
        plt.plot(line, line)
        plt.xlabel('Predicted' + unitsChar[metric])
        plt.ylabel('Measured' + unitsChar[metric])
        plt.title(metric)
        fig_plot.savefig('Comparison_plot'+metric+'.png')

        # arc_correction = Corrections["Arc Correction"].tolist()
        # dynamic_correction = Corrections["Dynamic Correction"].tolist()
        #
        # fig_surfplot = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(arc_correction, dynamic_correction, Y, color='red')
        # ax.scatter(arc_correction, dynamic_correction, predictions, color='blue')
        # ax.set_xlabel('Arc Correction')
        # ax.set_ylabel('Dynamic Correction')
        # ax.set_zlabel('Bead Width (mm)')
        # fig.legend(['Actual', 'Predicted'])
        # plt.show()
        # fig_surfplot.savefig('3D Plot '+ metric + '.png')


def classification_noRFE(Characterization, X, key):

    Y = Characterization[key]

    Y_range = np.amax(Y) - np.amin(Y)
    print(Y_range)

    classifier = RandomForestClassifier(n_estimators=1000)
    classifier.fit(X, Y)
    cv = LeaveOneOut()

    n_scores = cross_val_score(classifier, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
    predictions = cross_val_predict(classifier, X, Y, cv=cv, n_jobs=-1)

    print('Accuracy')
    print(n_scores)
    print('Average:')
    print(n_scores.mean())

    print(Y)
    print(predictions)


def classification_RFE(X, Y, metric, output_path):


    try:
        new_path = os.path.join(output_path, 'Random Forests')
        os.mkdir(new_path)
    except:
        os.chdir(new_path)


    X_new = X
    loo = LeaveOneOut()
    loo.get_n_splits(X_new)

    counter = 1

    X_features = X_new.columns.to_numpy()
    number_of_features = len(X_features)

    summary_df = pd.DataFrame(columns=['Number of Features', 'MAE', 'Least Important Feature'])
    all_num_features, all_mae, all_least_import = list(), list(), list()

    while number_of_features >= 7:

        y_pred, y_true = list(), list()

        for train_index, test_index in loo.split(X_new):
            X_train, X_test = X_new.iloc[train_index], X_new.iloc[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            classifier = RandomForestClassifier(n_estimators=1000, min_samples_split=3)

            classifier.fit(X_train, Y_train)
            y_hat = classifier.predict(X_test)
            y_pred.append(y_hat[0])
            y_true.append(Y_test[0])

        feat_imp = permutation_importance(classifier, X_train, Y_train)
        combined_feature_importance = feat_imp.importances_mean
        sorted_idx = combined_feature_importance.argsort()

        curr_score = accuracy_score(y_true, y_pred)

        file_suffix = metric + '_' + str(number_of_features) + "features"


        importances_df = pd.DataFrame(columns=['Feature', 'Importance'])
        importances_df['Feature'] = X_features
        importances_df['Importance'] = combined_feature_importance
        importances_df.to_csv(new_path + '\\Importances_ ' + file_suffix + '.csv')

        least_valuable_feature = X_features[sorted_idx[0]]
        X_new = X_new.drop(least_valuable_feature, axis=1)

        all_num_features.append(number_of_features)
        all_mae.append(curr_score)
        all_least_import.append(least_valuable_feature)

        new_score = accuracy_score(y_true, y_pred)
        print('Iteration %d......Current score: %.5f' % (counter, curr_score))
        print("Current Number of features %d" % number_of_features)
        X_features = X_new.columns.to_numpy()
        number_of_features = len(X_features)
        print('Least valuable feature: %s .... MSE: %.5f ' % (least_valuable_feature, new_score))
        print("New number of features %d" % number_of_features)

        counter = counter + 1

    summary_df['Number of Features'] = all_num_features
    summary_df['MAE'] = all_mae
    summary_df['Least Important Feature'] = all_least_import
    summary_df.to_csv(new_path + '\\Summary_ ' + file_suffix + '.csv')


def regression_RFE(Y, X, unitsChar,metric):
    X = dp.normalize_data(X)
    print('RFE Function Called')

    cwd = os.getcwd()

    try:
        new_path = os.path.join(cwd, metric)
        os.mkdir(new_path)
    except:
        os.chdir(new_path)

    X_new = X

    loo = LeaveOneOut()
    loo.get_n_splits(X_new)
    print('Leave-One-Out Splits Acquired')

    counter = 1

    X_features = X_new.columns.to_numpy()
    number_of_features = len(X_features)

    summary_df = pd.DataFrame(columns=['Number of Features', 'MAE', 'Least Important Feature'])
    all_num_features, all_mae, all_least_import = list(), list(), list()

    while number_of_features >= 7:
        print('RFE Loop Iteration: %d' % counter)

        y_pred, y_true = list(), list()

        print('Training loop begins')
        for train_index, test_index in loo.split(X_new):
            X_train, X_test = X_new.iloc[train_index], X_new.iloc[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            regressor = RandomForestRegressor(n_estimators=100, min_samples_split=3)

            regressor.fit(X_train, Y_train)
            y_hat = regressor.predict(X_test)
            y_pred.append(y_hat[0])
            y_true.append(Y_test[0])
        print('Training loop ends')
        feat_imp = permutation_importance(regressor, X_train, Y_train)
        combined_feature_importance = feat_imp.importances_mean
        sorted_idx = combined_feature_importance.argsort()

        curr_score = mean_absolute_error(y_true, y_pred)

        file_suffix = metric + '_' + str(number_of_features) + "features"

        line = np.linspace(min(y_true), max(y_true), 30)
        comp_plot = plt.figure()
        plt.scatter(y_true, y_pred)
        plt.plot(line, line)
        plt.xlabel('Measured' + unitsChar[metric])
        plt.ylabel('Predicted' + unitsChar[metric])
        comp_plot.savefig(new_path + '\\CompPlot_' + file_suffix + '.png')

        importances_df = pd.DataFrame(columns=['Feature', 'Importance'])
        importances_df['Feature'] = X_features
        importances_df['Importance'] = combined_feature_importance
        importances_df.to_csv(new_path + '\\Importances_ ' + file_suffix + '.csv')

        least_valuable_feature = X_features[sorted_idx[0]]
        X_new = X_new.drop(least_valuable_feature, axis=1)

        all_num_features.append(number_of_features)
        all_mae.append(curr_score)
        all_least_import.append(least_valuable_feature)

        new_score = mean_absolute_error(y_true, y_pred)
        print('Iteration %d......Current score: %.5f' % (counter, curr_score))
        print("Current Number of features %d" % number_of_features)
        X_features = X_new.columns.to_numpy()
        number_of_features = len(X_features)
        print('Least valuable feature: %s .... MSE: %.5f ' % (least_valuable_feature, new_score))
        print("New number of features %d" % number_of_features)

        counter = counter + 1

    summary_df['Number of Features'] = all_num_features
    summary_df['MAE'] = all_mae
    summary_df['Least Important Feature'] = all_least_import
    summary_df.to_csv(new_path + '\\Summary_ ' + file_suffix + '.csv')


def multimetric_regression(Characterization, X, unitsChar):

    for metric in Characterization:
        regression_RFE(Characterization[metric], X, unitsChar, metric)
