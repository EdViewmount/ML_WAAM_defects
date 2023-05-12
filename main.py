import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

import plotting

import random_forests as rf
import data_processing as dp
import segmentation as seg
import neural_network as nn

from tkinter import filedialog


def get_shape(NumPts,path,attribute):
    hfilename = "Height Profile Bead "
    wfilename = " Width Profile.csv"
    BeadShapes = []
    for i in range(1, NumPts + 1):
        tempBead = dp.BeadShape(i)
        if attribute == 'Height':
            dataHeight = pd.read_csv(path + '\\Height Profiles\\' + hfilename + str(i) + '.csv')
            X = dataHeight["X(mm)"].tolist()
            Z = dataHeight["Z(mm)"].tolist()
            tempBead.add_height(X, Z)
            tempBead.profile_trim(path, i)
            del dataHeight, X, Z
        elif attribute == 'Width':
            dataWidth = pd.read_csv(path + '\\Width and Centerline\\Bead' + str(i) + wfilename)
            L = dataWidth['Length'].tolist()
            Width = dataWidth['Width'].tolist()
            CenterLineDeviation = dataWidth['CenterLineDeviation']
            tempBead.add_width(L, Width,CenterLineDeviation)
            del dataWidth, L, Width, CenterLineDeviation

        BeadShapes.append(tempBead)

    return BeadShapes

#Main Program

#Enter main problem parameters
path = filedialog.askdirectory()
outputMainPath = os.path.join(path, 'Output')


attribute = 'Height'
metric = 'Std'
heightMetrics = ['Mean','Std','Peak to Valley']
widthMetrics = ['Mean','CenterLine']


NumPts = 43

num_windows = 10
percent_overlap = 0.15


##############################################
#Get Bead Shape Profiles
###########################################

BeadShapes = get_shape(NumPts, path, attribute)

##################################################################
# Extract from TDMS
##################################################################

#Extract settings from csv file
Settings = pd.read_csv(path + '\\Settings.csv')
#Extract time series data from tdms file
Beads = dp.extract_labview(path,NumPts)
#Extract IR data from csv files
Beads = dp.extract_IR_data(path,Beads,NumPts)
#Extract feature units
units = pd.read_csv(path + '\\feature units.csv')

#################################
#Denoise Waveforms
######################################

for bead in Beads:
    #bead.denoise('lemData','Welding Voltage')
    #bead.denoise('lemData', 'Welding Current')
    bead.denoise('audio', 'Audio')

#Plot bead profiles prior to pre-processing
for beadshape in BeadShapes:
    plotting.plot_profile(beadshape,'Height Profile',outputMainPath)


############################################################
#Pre-processing
###########################################################

for bead in Beads:
    i = bead.number
    print(i)

    bead.add_settings(Settings.iloc[i-1,:])
    bead.remove_prepost_time()

    if attribute == 'Height':
        xtrim, xendtrim = BeadShapes[i - 1].trim_slopes('Height', 'x')
        bead.trim_profile_time(xtrim, xendtrim)


#########################################
#Plotting
#########################################

#Plot bead profiles after trimming
for beadshape in BeadShapes:
    plotting.plot_profile(beadshape,'Trimmed Height Profile',outputMainPath)

for bead in Beads:
    plotting.plot_spectrograms(bead,outputMainPath, percent_overlap, num_windows, sr = 22050, group = 'audio', waveform = 'Audio')
    plotting.plot_spectrograms(bead, outputMainPath, percent_overlap, num_windows, sr=20000, group='lemData', waveform='Welding Voltage')
    plotting.plot_spectrograms(bead, outputMainPath, percent_overlap, num_windows, sr=20000, group='lemData', waveform='Welding Current')


for bead in Beads:
    plotting.plot_timeseries(bead,outputMainPath,'weldData','lemData','audio','meltTemps')

# ############################################
# # Create data frame and output dictionary
# ##########################################

outputName = attribute + ' ' + metric

try:
    outputPath = os.path.join(outputMainPath, outputName)
    os.mkdir(outputPath)
except:
    pass

X,Y, Segments = seg.segment_assemble(Beads, BeadShapes, num_windows, percent_overlap, attribute = attribute, metric = metric)
X = X.drop('Wire Feed Speed Kurt',axis = 1)

#Plot feature correlations with respect to output
for (columnName, columnData) in X.iteritems():
    plotting.plot_segmentXY(X, Y, columnName, outputName, outputPath,units)

#Obtain basic stats about output metric
Y = np.array(Y)
Y_stats = pd.DataFrame(columns = ['Mean','Range','Max','Min'])
Y_stats['Mean'] = [np.mean(Y)]
Y_stats['Range'] = [max(Y) - min(Y)]
Y_stats['Max'] = [max(Y)]
Y_stats['Min'] = [min(Y)]
Y_stats.to_csv(outputPath + '\\output_stats.csv')

#Plot Feature Distributions
for (columnName, columnData) in X.iteritems():

    try:
        dist_path = os.path.join(outputMainPath, 'Feature Distributions')
        os.mkdir(dist_path)
    except:
        pass
    plotting.plot_histogram(columnData,columnName,dist_path)

print('Output %s ranges from %.6f to %.6f. The range is %.6f' % (outputName,min(Y), max(Y), (max(Y)-min(Y))))


# # ######################################
# # #Eliminate Highly Correlated Features
# # #######################################

#Compute correlation matrix
corr = X.corr()
plt.rcParams.update({'font.size': 27})
fig_corr = plt.figure(figsize = [35,35])
ax = fig_corr.add_subplot()
sns.heatmap(corr , ax = ax)
fig_corr.figure.savefig(outputMainPath+'\\Correlation_Matrix.png')

columns = np.full((corr.shape[0],), True, dtype=bool)
cols = X.columns.tolist()


# Remove one of every pair of columns that are 95% correlated
print("Dropping data points that are 95% correlated to existing data:")
for i in range(corr.shape[0]):
    if i > 0 :
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= 0.95:
                if columns[i] and columns[j]:
                    print(str(cols[j]) + " " + str(j) + ": (95%+ correlated to " + str(cols[i]) + " " + str(i) + ")")
                    X = X.drop(str(cols[j]), axis = 1)
                    columns[j] = False
                elif columns[j]:
                    columns[j] = True

# ############################
# ##Machine Learning Model ###
# ############################
n,bins = plotting.plot_histogram(Y,outputName, outputPath)

del Beads

#Run model

modelType = 'LSTM'
nn_epochs = 1600
classify_epochs = 50
lstm_epochs = 10


if modelType == 'LSTM':
    segsPerBead = 10
    numBeads = NumPts
else:
    segsPerBead = None
    numBeads = None
print('Model Initializing')


try:
    ml_outputPath = os.path.join(outputPath, modelType)
    os.mkdir(ml_outputPath)
except:
    pass

if modelType == 'RF':
    Y_pred = rf.regression_RFE(X, Y, ml_outputPath, outputName)
else:
    Y_pred = nn.neuralNetworkMain(X, Y, ml_outputPath, modelType=modelType, epochs=nn_epochs, lr=1e-4,
                                   foldSplits = 25, numBeads = numBeads, segsPerBead = segsPerBead)

BeadShapes = seg.prediction_assignment(Segments,BeadShapes, Y_pred)

for beadshape in BeadShapes:
    plotting.plot_output(beadshape,ml_outputPath,attribute,metric)


main(path, 'Peak to Valley', 'Height')


# for metric in widthMetrics:
#     main(path,metric,'Width')

# for modelType in Models:
#
#     nn_epochs = 1000
#
#     if modelType == 'LSTM':
#         segsPerBead = 10
#         numBeads = NumPts
#     else:
#         segsPerBead = None
#         numBeads = None
#     print('Model Initializing')
#
#
#     try:
#         ml_outputPath = os.path.join(outputPath, modelType)
#         os.mkdir(ml_outputPath)
#     except:
#         pass
#
#
#     if modelType  ==  'RF':
#         Y_pred = rf.regression_RFE(X, Y, ml_outputPath, outputName)
#     else:
#         Y_pred = nn.neuralNetworkMain(X, Y, ml_outputPath, modelType=modelType, epochs=nn_epochs, lr=1e-4,
#                                       foldSplits=23, numBeads=numBeads, segsPerBead=segsPerBead)
#
#     BeadShapes = seg.prediction_assignment(Segments,BeadShapes, Y_pred)
#
#     for beadshape in BeadShapes:
#         plotting.plot_output(beadshape,outputPath,attribute,metric)