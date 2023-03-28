import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

# def main():

#Main Program

path = filedialog.askdirectory()

units_ts = {'Welding Voltage': '(V)', 'Welding Current': '(A)', 'Wire Feed Speed': '(in/min)', 'Travel Speed': '(in/min)'}
unitsChar = {'Bead Width': '(mm)','Bead Height': '(mm)', 'Ra': '(um)','Rz':'(um)','Bead Height Standard Deviation': '(mm)'}
NumPts = 45
Xunits = pd.read_csv(path+'\\Xunits.csv')

##############################################
# Bead Shape Profiles
###########################################

hfilename = "Height Profile Bead "
wfilename = " Width Profile.csv"
BeadShapes = []
for i in range(1, NumPts + 1):

    dataHeight = pd.read_csv(path+'\\Height Profiles\\' + hfilename + str(i) + '.csv')
    # dataWidth = pd.read_csv(path + '\\Width Profiles\\Bead' + str(i) + wfilename)
    # L = dataWidth['Length'].tolist()
    # Width = dataWidth['Width'].tolist()
    # CtrLineDeviation = dataWidth['CenterLineDeviation']
    X = dataHeight["X(mm)"].tolist()
    Z = dataHeight["Z(mm)"].tolist()
    del dataHeight
    tempBead = dp.BeadShape(i,X,Z)
    tempBead.profile_trim(path,i)
    BeadShapes.append(tempBead)
    del tempBead,X,Z

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
Beads = dp.extract_tdms(path,NumPts)
#Extract IR data from csv files
Beads = dp.extract_IR_data(path,Beads,NumPts)

for bead in Beads:
    i = bead.number
    bead.add_settings(Settings.iloc[i-1,:])
    #bead.filter_blips()
    bead.remove_prepost_time()


#################################
# Get Audio Features
######################################

Beads = af.file_extract(NumPts,path,Beads)
Beads = af.denoise_audio(Beads)

# ############################################
# # Create data frame and output dictionary
# ##########################################

# # X, Characterization = gm.structure_data(WeldData,LEMData, Beads, Audio, path)

#Remove problematic sample from data set
Beads.pop(28)
BeadShapes.pop(28)
NumPts = NumPts - 1
X,Y = seg.segment_assemble(Beads, BeadShapes, 10, 0.15)
#X.drop(['Melt Pool Temperature Kurt','Melt Pool Temperature Std'],axis=1)
# # plotting.plot_segmentXY(X,Y,'Welding Current Mean',Xunits)

# ######################################
# #Eliminate Highly Correlated Features
# #######################################

fig_corr = plt.figure()
corr = X.corr()
plt.subplots(figsize=(30,25))
sns.heatmap(corr)
fig_corr.savefig('Correlation_Matrix.png')

columns = np.full((corr.shape[0],), True, dtype=bool)
cols = X.columns.tolist()
# Remove one of every pair of columns that are 95% correlated
print("Dropping data points are 95% correlated to existing data:")
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

############################
##Machine Learning Model ###
############################

# # Y = Characterization['Porosity']
# # por_idx = [i for i in range(len(Y)) if Y[i] == 1]
# # X = X.drop(por_idx)

print('Model Initializing')
#rf.regression_RFE(Y, X, unitsChar,'Bead Height')
#nn.neuralNetwork(X,Y)
nn.recurrentNeuralNetwork(X,Y,NumPts,8)

# # # # if __name__ == "__main__":
# # # #     print("Running")
# # # #     main()
# #