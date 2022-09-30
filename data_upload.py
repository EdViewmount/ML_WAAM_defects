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
        LEMGroup = temp_file['LEM Box']

        for channel in RobGroup.channels():
            TempRobData[channel.name] = channel[:]

        for channel in LEMGroup.channels():
            TempWeldData[channel.name] = channel[:]

        for channel in WeldGroup.channels():
            TempWeldData[channel.name] = channel[:]

        AllRobData["Bead" + str(i)] = TempRobData
        AllWeldData["Bead" + str(i)] = TempWeldData


    return AllRobData, AllWeldData


def extract_IR_profiles(path,num):

    IR_profiles = {}

    for i in range(1,num):

        if i < 10:
            beadnumstr = "0" + str(i)
        else:
            beadnumstr = str(i)

        filename = 'Bead' + beadnumstr + '_basetemp'

        IR_profiles['Bead'+str(i)] = np.genfromtxt(path+filename+'.csv', delimiter = ',')

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
