import os
from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import plotting
import audio_features as af
import random_forests as rf
import data_upload as du


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


def difference(x):
    i=0
    diff=[]

    for i in range(0,len(x),5):

        if i < len(x) - 5:
            diff.append(x[i + 5] - x[i])
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


def calculate_energy(Data,path):
    i=1

    Energy = {}
    Energy = pd.DataFrame(Energy)
    plt.figure()

    for key in Data:

        time = Data[key]['Time']
        volt = Data[key]['Welding Voltage']
        curr = Data[key]['Welding Current']
        #wfs = Data[key]['Wire Feed Speed']
        energy_per_in = (volt * curr) / (25* (25.4/60))

        energy_temp = {}

        energy_temp[key] = energy_per_in
        energy_temp = pd.DataFrame.from_dict(energy_temp)


        if i == 1:
            Energy = energy_temp
        else:
            Energy = pd.concat([Energy, energy_temp],axis=1)
        i=i+1

        plt.plot(time,energy_per_in)
        plt.xlabel('Time (s)')
        plt.ylabel('Energy Input (J/mm)')

        del energy_temp

    plt.savefig(path + '\\Energy Input\\energy_input.png')
    EnergyDF = pd.DataFrame.from_dict(Energy)
    pd.DataFrame.to_csv(EnergyDF,path+'\\Energy Input\\Energy_Input.csv')
    return Energy


def control_chart(data,windows):
    data_length = len(data)
    window_length = data_length/windows
    data_new = np.reshape(data,[-1,window_length])
    mean_data = np.mean(data_new,axis=1)


def filter_blips():
    for key in RobData:
        xtemp = RobData[key]['Robot X']
        filter_idx = np.where(xtemp > 500)

        RobData[key]['Robot X'] = np.delete(xtemp, filter_idx)
        RobData[key]['Robot Y'] = np.delete(RobData[key]['Robot Y'], filter_idx)
        RobData[key]['Robot Z'] = np.delete(RobData[key]['Robot Z'], filter_idx)
        RobData[key]['Time'] = np.delete(RobData[key]['Time'], filter_idx)

        WeldData[key]['Welding Voltage'] = np.delete(WeldData[key]['Welding Voltage'], filter_idx)
        WeldData[key]['Welding Current'] = np.delete(WeldData[key]['Welding Current'], filter_idx)
        WeldData[key]['Wire Feed Speed'] = np.delete(WeldData[key]['Wire Feed Speed'], filter_idx)
        WeldData[key]['Time'] = np.delete(WeldData[key]['Time'], filter_idx)


#Main Program
path = 'C:\\Users\\emiramon\\Documents\\Data\\Dataset 3\\'
units_ts = {'Welding Voltage': '(V)', 'Welding Current': '(A)', 'Wire Feed Speed': '(in/min)', 'Travel Speed': '(in/min)'}
unitsChar = {'Bead Width': '(mm)','Bead Height': '(mm)', 'Ra': '(um)','Rz':'(um)','Bead Height Standard Deviation': '(mm)'}

Xunits = pd.read_csv(path+'Xunits.csv')


# def main():

NumPts = 46
##############################################
# Compute bead height standard deviation
###########################################


file = "Height Profile Bead "
extension = ".csv"

Beads = {}
for i in range(1, NumPts + 1):
    data = pd.read_csv(path+'Height Profiles\\' + file + str(i) + extension)
    X = data["X(mm)"].tolist()
    Z = data["Z(mm)"].tolist()
    temp = du.BeadHeight(X, Z)
    Beads["Bead" + str(i)] = temp

for key in Beads:
    Beads[key] = du.profile_trim(Beads[key])

Stdevs = []

for key in Beads:
    ztemp = np.array(Beads[key].z)
    zstdev = np.std(ztemp[np.isfinite(ztemp)])
    Stdevs.append(zstdev)

###################################################################
#Get Bead characterization metrics
###################################################################
Characterization = pd.read_csv(path+'Characterization.csv')
Characterization['Bead Height Standard Deviation'] = Stdevs
Characterization.to_csv(path+'Characterization.csv', index=False)
# Characterization = Characterization.drop(['Bead','Dynamic Correction','Arc Correction','Sa'], axis=1)
Characterization = Characterization.drop(['Bead'], axis=1)
Settings = pd.read_csv(path+ 'Process Settings.csv')
##################################################################
# Extract from TDMS
##################################################################

path2 = path+'Bead'
RobData, WeldData = du.extract_tdms_wlem(path2, NumPts)
IR_profiles =du.extract_IR_profiles(path + 'IR Data\\Line Profiles\\',NumPts)

RobData = calculate_speed(RobData)

for key in WeldData:
    WeldData[key]['Travel Speed'] = RobData[key]['Travel Speed']

basetemp = []
for key in IR_profiles:
    temp_ir = IR_profiles[key]
    basetemp.append(np.mean(temp_ir))

#################################
# Get Audio Features
######################################

Audio, SR = af.file_extract(NumPts, path2)

#Denoise Audio
af.denoise_audio(Audio)

AudioFrame = {}
sr = SR['Bead1']
for key in Audio:
    dfkey = key + ' features'
    AudioFrame[dfkey] = af.extract_basic_features(Audio[key], 48000)

############################################
# Create data frame
##########################################

WeldAvg = get_averages(WeldData)
X = create_dataframe(WeldAvg, AudioFrame)
X["Base Temperature"] = basetemp


# plotting.parameter_correlations(X,Characterization,Xunits,unitsChar)
######################################
#Eliminate Highly Correlated Features
#####################################
#
# fig_corr = plt.figure()
# corr = X.corr()
# plt.subplots(figsize=(30,25))
# sns.heatmap(corr)
# plt.show()
# fig_corr.savefig('Correlation_Matrix.png')
#
# columns = np.full((corr.shape[0],), True, dtype=bool)
# cols = X.columns.tolist()
# # Remove one of every pair of columns that are 95% correlated
# print("Dropping data points are 95% correlated to existing data:")
# for i in range(corr.shape[0]):
#     if i > 0 :
#         for j in range(i+1, corr.shape[0]):
#             if corr.iloc[i,j] >= 0.95:
#                 if columns[i] and columns[j]:
#                     print(str(cols[j]) + " " + str(j) + ": (95%+ correlated to " + str(cols[i]) + " " + str(i) + ")")
#                     X = X.drop(str(cols[j]), axis = 1)
#                     columns[j] = False
#                 elif columns[j]:
#                     columns[j] = True


############################
##Random Forest Model ###
############################

# Y = Characterization['Porosity']
#
# por_idx = [i for i in range(len(Y)) if Y[i] == 1]
# X = X.drop(por_idx)

# Characterization = Characterization.drop(por_idx)
# Characterization = Characterization.drop(['Porosity','Bead Width', 'Bead Height'], axis = 1)
# rf.regression_noRFE(Characterization, X, unitsChar)


plt.ioff()


def spectra_allbeads(welddata, audio, cut_freq):

    cwd = os.getcwd()
    new_path = cwd + '\\Dataset 3\\Frequency Plots '

    frequency_table = pd.DataFrame(columns=['Bead', 'Audio Max Frequency', 'Voltage Max Frequency', 'Current Max Frequency'])
    audio_maxfreq = []
    volt_maxfreq = []
    curr_maxfreq = []

    try:
        os.mkdir(new_path)
    except:
        os.chdir(new_path)

    for i in range(1, NumPts + 1):
        Bead = 'Bead' + str(i)
        volt_plot, volt_freq, volt_mag = plotting.plot_data_spectra(welddata, Bead, 'Welding Voltage', cut_freq)
        curr_plot, curr_freq, curr_mag = plotting.plot_data_spectra(welddata, Bead, 'Welding Current', cut_freq)
        audio_plot, audio_freq, audio_mag = af.plot_audio_spectra(audio, 48000, Bead, cut_freq)

        audio_maxfreq_temp = audio_freq[np.argmax(audio_mag)]
        volt_maxfreq_temp = volt_freq[np.argmax(volt_mag)]
        curr_maxfreq_temp = curr_freq[np.argmax(curr_mag)]
        audio_maxfreq.append(audio_maxfreq_temp)
        volt_maxfreq.append(volt_maxfreq_temp)
        curr_maxfreq.append(curr_maxfreq_temp)

        audio_plot.savefig('Audio Frequency Plot Bead' + str(i) + '.png')
        volt_plot.savefig('Voltage Frequency Plot Bead' + str(i) + '.png')
        curr_plot.savefig('Current Frequency Plot Bead' + str(i) + '.png')

    frequency_table['Bead'] = np.arange(1, NumPts + 1)
    frequency_table['Audio Max Frequency'] = audio_maxfreq
    frequency_table['Voltage Max Frequency'] = volt_maxfreq
    frequency_table['Current Max Frequency'] = curr_maxfreq

    frequency_table.to_csv('Max Frequencies.csv', index=False)




# #
# if __name__ == "__main__":
#     print("Running")
#     main()
#

