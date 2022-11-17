from nptdms import TdmsFile
import numpy as np
import pandas as pd


class BeadHeight:

    def __init__(self, x, z):
        self.x = x
        self.z = z


def extract_tdms_data(path,NumPts):

    AllRobData = {}
    AllWeldData = {}

    for i in range(1, NumPts+1):

        TempRobData = {}
        TempWeldData = {}

        if i < 10:
            beadnumstr = "0" + str(i)
        else:
            beadnumstr = str(i)

        temp_file = TdmsFile.read(path + beadnumstr + '\\Bead01.tdms')

        RobGroup = temp_file["Robot Data"]
        WeldGroup = temp_file["Welding Data"]

        for channel in RobGroup.channels():
            TempRobData[channel.name] = channel[:]

        for channel in WeldGroup.channels():
            TempWeldData[channel.name] = channel[:]

        AllRobData["Bead" + str(i)] = TempRobData
        AllWeldData["Bead" + str(i)] = TempWeldData

    return AllRobData, AllWeldData


def extract_tdms_wlem(path,NumPts):

    AllRobData = {}
    AllWeldData = {}
    AllLEMData ={}

    for i in range(1, NumPts+1):

        TempRobData = {}
        TempWeldData = {}
        TempLEMData = {}

        if i < 10:
            beadnumstr = "0" + str(i)
        else:
            beadnumstr = str(i)

        temp_file = TdmsFile.read(path + beadnumstr + '\\Bead01.tdms')

        RobGroup = temp_file["Robot Data"]
        WeldGroup = temp_file["Welding Data"]
        LEMGroup = temp_file['LEM Box']

        for channel in RobGroup.channels():
            TempRobData[channel.name] = channel[:]

        for channel in LEMGroup.channels():
            TempLEMData[channel.name] = channel[:]

        for channel in WeldGroup.channels():
            TempWeldData[channel.name] = channel[:]

        LEMtime = get_LEMtime(TempWeldData)
        TempWeldData['Time'] = LEMtime

        AllRobData["Bead" + str(i)] = TempRobData
        AllWeldData["Bead" + str(i)] = TempWeldData
        AllLEMData["Bead" + str(i)] = TempLEMData

    return AllRobData, AllWeldData, AllLEMData


def get_LEMtime(data):

    sample_rate = 20000

    volt_temp = data['Welding Voltage']
    n_samples = len(volt_temp)

    dt = n_samples/sample_rate

    time = []

    for i in range(0,n_samples):
        time.append(dt*(i+1))

    return time


def extract_IR_profiles(path,num):

    IR_profiles = {}

    for i in range(1,num+1):

        if i < 10:
            beadnumstr = "0" + str(i)
        else:
            beadnumstr = str(i)

        filename = 'Bead' + beadnumstr + '_basetemp'

        temp_df = pd.read_csv(path+filename+'.csv')

        IR_profiles['Bead'+str(i)] = temp_df['Bead' + beadnumstr + '.seq:Line 1 [C]:mean:vert'].values

    return IR_profiles


def profile_trim(bead):

    z = bead.z
    z = np.array(z)
    endpoints = []
    for i in range(0, len(z)-1):
        if z[i] > 0.65*np.nanmedian(z):
            endpoints.append(i)
    endpoints = np.array(endpoints)
    start = endpoints[0]
    end = endpoints[len(endpoints) - 1]
    bead.z = z[start:end]
    bead.x = bead.x[start:end]
    return bead

def profile_trim(bead):

    z = bead.z
    x = bead.x

    x = np.array(x)
    z = np.array(z)
    endpoints = []

    start_idx= np.where(x < 12.7 )
    end_idx = np.where(x > 87.3)

    filter_idx = np.concatenate((start_idx,end_idx), axis = None)

    z = np.delete(z, [filter_idx])
    x = np.delete(x, [filter_idx])

    bead.z = z
    bead.x = x

    return bead
