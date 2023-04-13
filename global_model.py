import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import plotting
import audio_features as af
import data_processing as dp
import scipy as sc


def get_averages(dictionary):

    dict_avg = {}

    for outer_key in dictionary:

        dict_avg[outer_key] = {}

        for inner_key in dictionary[outer_key]:

            if inner_key != 'Time':
                x = dictionary[outer_key][inner_key]

                dict_avg[outer_key][inner_key + ' Mean'] = [np.mean(x[np.isfinite(x)])]
                dict_avg[outer_key][inner_key + ' Std'] = [np.std(x[np.isfinite(x)])]

    return dict_avg


def create_dataframe(WeldAvg, AudioFrames):

    i = 1
    for key in WeldAvg:

        dtemp = WeldAvg[key]

        if i == 1:
            DF = pd.DataFrame.from_dict(dtemp, orient='columns')
        else:
            df_temp = pd.DataFrame.from_dict(dtemp, orient='columns')
            DF = pd.concat([DF, df_temp], axis=0, ignore_index=True)

        i = i + 1

    DF = DF.drop(['Motor Current Mean','Motor Current Std','Travel Speed Std',], axis=1)

    i = 1

    for key in AudioFrames:
        d = af.feature_averages(AudioFrames[key])
        if i == 1:
            audio_df = pd.DataFrame.from_dict(d, orient='columns')
        else:
            dftemp = pd.DataFrame.from_dict(d, orient='columns')
            audio_df = pd.concat([audio_df, dftemp], ignore_index=True, axis=0)

        i = i + 1


    DF = pd.concat([DF, audio_df], axis=1)
    DF = DF.drop(['Tempo Std', 'Tempo Skew', 'Tempo Kurt'], axis=1)

    return DF

#
# def get_characterization_metrics(path, Beads):
#
#     Stdevs = height_stdevs(Beads)
#     Characterization = pd.read_csv(path + 'Characterization.csv')
#     Characterization['Bead Height Standard Deviation'] = Stdevs
#     Characterization.to_csv(path + 'Characterization.csv', index=False)
#     # Characterization = Characterization.drop(['Bead','Dynamic Correction','Arc Correction','Sa'], axis=1)
#     Characterization = Characterization.drop(['Bead'], axis=1)
#
#     return Characterization


# def structure_data(WeldData,LEMData, Beads, Audio, path):
#
#     Characterization = get_characterization_metrics(path, Beads)
#     WeldData = dp.merge_data_dicts(LEMData, WeldData)
#
#     #################################
#     # Get Audio Features
#     ######################################
#
#     AudioFrame = {}
#     for key in Audio:
#         dfkey = key
#         AudioFrame[dfkey] = af.extract_basic_features(Audio[key], 48000)
#
#     ############################################
#     # Create data frame
#     ##########################################
#
#     WeldAvg = get_averages(WeldData)
#     X = create_dataframe(WeldAvg, AudioFrame)
#
#     return X, Characterization

############################################################################
#Object Oriented Functions
###########################################################################

def computeOutputs(BeadShapes, attribute, metric):
    Y = []

    for beadshape in BeadShapes:
        profile = getattr(beadshape,attribute)

        if metric == 'Mean':
            y = np.mean(profile, axis=1)

        elif metric == 'Std':

            y = np.std(profile[np.isfinite(profile)])

        elif metric == 'Peak to Valley':

            peak2Val= beadshape.segment(0.15,10,attribute = attribute,metric = 'Peak to Valley')
            y = np.mean(peak2Val)

        Y.append(y)
    return Y


def get_averages_object(bead, attribute):

    dict_avg = {}

    data = getattr(bead,attribute)

    for inner_key in data:

        if inner_key != 'Time':
            x = data[inner_key]

            dict_avg[inner_key + ' Mean'] = [np.mean(x[np.isfinite(x)])]
            dict_avg[inner_key + ' Std'] = [np.std(x[np.isfinite(x)])]
            dict_avg[inner_key + ' Kurt'] = sc.stats.kurtosis(data, axis=1)

            setattr(bead,attribute+'Stats',dict_avg)

    return bead


def create_dataframe_object(Beads):

    i = 1
    for bead in Beads:

        dtemp = bead.allBeadStats
        audio_df = bead.audioFrames

        if i == 1:
            DF = pd.DataFrame.from_dict(dtemp, orient='columns')
        else:
            df_temp = pd.DataFrame.from_dict(dtemp, orient='columns')
            DF = pd.concat([DF, df_temp], axis=0, ignore_index=True)

        i = i + 1

        DF = pd.concat([DF, audio_df], axis=1)
        DF = DF.drop(['Tempo Std', 'Tempo Skew', 'Tempo Kurt'], axis=1)

    DF = DF.drop(['Motor Current Mean','Motor Current Std','Travel Speed Std',], axis=1)

    return DF


def structure_data_object(Beads, BeadShape, attribute, metric):

    Y = computeOutputs(BeadShape,attribute,metric)

    for bead in Beads:
        bead = get_averages_object(bead, 'lemData')
        bead = get_averages_object(bead, 'weldData')
        bead = get_averages_object(bead, 'meltPoolTemps')
        bead.merge_data()

        #Get audio data frame and store
        bead.audioFrames = af.extract_basic_features(bead.audio['Audio'], 48000)

    ############################################
    # Create data frame
    ##########################################

    X = create_dataframe(Beads)

    return X, Y