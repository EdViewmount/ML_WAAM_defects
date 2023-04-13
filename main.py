import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

import plotting
import audio_features as af
import random_forests as rf
import data_processing as dp
import segmentation as seg
import neural_network as nn
import global_model as gm

from tkinter import *
from tkinter import filedialog


def difference(x):
    i=0
    diff=[]

    for i in range(0,len(x),4):

        if i < len(x) - 4:
            diff.append(x[i + 4] - x[i])
    diff = np.array(diff)

    return diff


def down_sample(t, n):
    i=0
    t_new = []

    for i in range(0,len(t), n):

        if i < len(t) - n:
            t_new.append(t[i])

    t_new = np.array(t_new)

    return t_new


def calculate_speed(RobotData):

    for key in RobotData:
        tempdata = RobotData[key]
        t = tempdata['Time']
        x = tempdata["Robot X"]
        y = tempdata["Robot Y"]
        z = tempdata["Robot Z"]

        dt = difference(t)
        dx = difference(x)
        dy = difference(y)
        dz = difference(z)

        vx = np.zeros(dx.shape, float)
        vy = np.zeros(dy.shape, float)
        vz = np.zeros(dz.shape, float)

        vx = dx/dt
        vy = dy / dt
        vz = dz / dt

        v_mag = np.sqrt(vx**2 + vy**2 + vz**2) *(60/24.5)

        np.size(v_mag)

        tempdata["Travel Speed"] = v_mag

        RobotData[key] = tempdata

    return RobotData


def on_click(text):

    #mode.set(text)

    return 0;


def output_classify(Y, cutoff):
    booleanArray = []

    for y in Y:

        if y > cutoff:
            booleanArray.append(1)
        else:
            booleanArray.append(0)

    return booleanArray


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
            CtrLineDeviation = dataWidth['CenterLineDeviation']
            tempBead.add_width(L, Width)
            del dataWidth, L, Width, CtrLineDeviation

        BeadShapes.append(tempBead)

    return BeadShapes

# def main():

#Main Program

#Xunits = pd.read_csv(path+'\\Xunits.csv')

#Enter main problem parameters
path = filedialog.askdirectory()
NumPts = 43
outputMainPath = os.path.join(path, 'Output')
num_windows = 10
percent_overlap = 0.15
attribute = 'Height'
metric = 'Std'

##############################################
#Get Bead Shape Profiles
###########################################

BeadShapes = get_shape(NumPts, path, attribute)

# mode = StringVar()
# win = Tk()
# b1= Button(win, text= "Plotting", command = on_click("Plotting")).pack()
# b2= Button(win, text= "Modeling", command = on_click("Modeling")).pack()

##################################################################
# Extract from TDMS
##################################################################

#Extract settings from csv file
Settings = pd.read_csv(path + '\\Settings.csv')
#Extract time series data from tdms file
Beads = dp.extract_labview(path,NumPts)
#Extract IR data from csv files
Beads = dp.extract_IR_data(path,Beads,NumPts)

#################################
#Denoise Waveforms
######################################

for bead in Beads:
    #bead.denoise('lemData','Welding Voltage')
    #bead.denoise('lemData', 'Welding Current')
    bead.denoise('audio', 'Audio')

#########################################
#Plotting
#########################################

#Plot bead profiles prior to pre-processing
# for beadshape in BeadShapes:
#     plotting.plot_profile(beadshape,'Height Profile',outputMainPath)

# for bead in Beads:
#     plotting.plot_spectrograms(bead,outputMainPath, percent_overlap, num_windows, sr = 22050, group = 'audio', waveform = 'Audio')
    #plotting.plot_spectrograms(bead, outputMainPath, percent_overlap, num_windows, sr=20000, group='lemData', waveform='Welding Voltage')
    #plotting.plot_spectrograms(bead, outputMainPath, percent_overlap, num_windows, sr=20000, group='lemData', waveform='Welding Current')

############################################################
#Pre-processing
###########################################################

for bead in Beads:
    i = bead.number
    bead.add_settings(Settings.iloc[i-1,:])
    bead.remove_prepost_time()
    xtrim, xendtrim = BeadShapes[i-1].trim_slopes('Height','x')
    bead.trim_profile_time(xtrim,xendtrim)

# #Plot bead profiles after trimming
# for beadshape in BeadShapes:
#     plotting.plot_profile(beadshape,'Trimmed Height Profile',outputMainPath)

# ############################################
# # Create data frame and output dictionary
# ##########################################

# # X, Characterization = gm.structure_data(WeldData,LEMData, Beads, Audio, path)WHI

outputName = attribute + ' ' + metric

try:
    outputPath = os.path.join(outputMainPath, outputName)
    os.mkdir(outputPath)
except:
    pass


X,Y, Segments = seg.segment_assemble(Beads, BeadShapes, num_windows, percent_overlap, attribute = attribute, metric = metric)

# for (columnName, columnData) in X.iteritems():
#
#     try:
#         dist_path = os.path.join(outputMainPath, 'Feature Distributions')
#         os.mkdir(dist_path)
#     except:
#         pass
#     plotting.plot_histogram(columnData,columnName,dist_path)

print('Output %s ranges from %.6f to %.6f. The range is %.6f' % (outputName,min(Y), max(Y), (max(Y)-min(Y))))


# # ######################################
# # #Eliminate Highly Correlated Features
# # #######################################

corr = X.corr()
fig_corr = sns.heatmap(corr)
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

# if mode == 'Plotting':
#     v =  1
# elif mode == 'Modeling':
#     v = 2

# ############################
# ##Machine Learning Model ###
# ############################

# # # Y = Characterization['Porosity']
# # # por_idx = [i for i in range(len(Y)) if Y[i] == 1]
# # # X = X.drop(por_idx)
# #Y_std = seg.output_array(BeadShapes,'z','Std', percent_overlap,num_windows)
# # plt.figure()
# # plt.scatter(Y,Y_std)

# #Create classification problem
Y = np.array(Y)
n,bins = plotting.plot_histogram(Y,outputName, outputPath)

#freqSplitIdx = np.where(n == 20)
binCutoff = 0.22593
boolY = output_classify(Y, binCutoff)
n_true = sum(boolY)
p_true = n_true/len(boolY)
print('Percent of Y that is true: %.3f' % p_true)
classifyDataframe = pd.DataFrame()
classifyDataframe['Peak to Valley (mm)'] = Y
classifyDataframe['>  0.1872'] = boolY
boolY = np.array(boolY)

del Beads

#Run model

modelType = 'NN Classify'
nn_epochs = 700
classify_epochs = 50
lstm_epochs = 1000
print('Model Initializing')
#rf.regression_RFE(Y, X, unitsChar,'Bead Height')
#Y_pred = rf.classification_RFE(X, boolY,outputName,outputPath)
#Y_pred = nn.neuralNetwork(X,Y, outputPath)
#Y_pred = nn.neuralNetwork_classify(X,boolY,outputPath)
#Y_pred = nn.recurrentNeuralNetwork(X,Y,NumPts,10,outputPath)


Y_pred = nn.neuralNetworkMain(X, Y, outputPath, modelType = modelType, epochs = nn_epochs ,lr = 1e-4,
                              foldSplits = 15, numBeads = None, segsPerBead = None)

# # # # if __name__ == "__main__":
# # # #     print("Running")
# # # #     main()
