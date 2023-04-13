from nptdms import TdmsFile
import numpy as np
import pandas as pd
import math
from math import floor
import segmentation as seg
import os
import librosa

from skimage.restoration import denoise_wavelet

import audio_features as af


class BeadShape:

    def __init__(self,beadNum):
        self.beadNum = beadNum
        self.x = None
        self.Height = None
        self.L = None
        self.Width = None
        self.centerLineDeviation = None
        self.peaktoval = None
        self.std = None
        self.numWindows = None
        self.overlap = None
        self.xtrim = None

    def add_height(self,X,Height):
        self.x= np.array(X)
        self.Height = np.array(Height)

    def add_width(self,L,Width,centerLineDeviation):
        self.L = L
        self.Width = np.array(Width)
        self.centerLineDeviation = []


    def profile_trim(self,path,beadNum,attributeX = 'x',attributeProfile = 'Height'):

        x = np.array(getattr(self,attributeX))
        z = np.array(getattr(self,attributeProfile))

        startEndPts = pd.read_csv(path + '\\Start end points.csv')
        start = startEndPts['Start'][beadNum - 1]
        end = startEndPts['End'][beadNum - 1]

        start_idx = np.where(x < start)
        end_idx = np.where(x > end)

        filter_idx = np.concatenate((start_idx, end_idx), axis=None)

        z = np.delete(z, [filter_idx])
        x = np.delete(x, [filter_idx])

        x = x - min(x)

        setattr(self,attributeX,x)
        setattr(self,attributeProfile,z)

    def segment(self,overlap,num_windows,attribute = 'Height',metric = 'Mean'):
        temp = getattr(self,attribute)
        temp = temp[np.isfinite(temp)]
        L = temp.size
        lw = seg.calculate_winlength(L, overlap, num_windows)
        windowedProfile = seg.segment_axis(temp, lw, percent_overlap=overlap, end="cut")
        #windowedProfile = np.delete(windowedProfile, 0, 0)

        setattr(self,attribute+ ' Windows',windowedProfile)

        if metric == 'Mean':
            y = np.mean(windowedProfile, axis=1)

        elif metric == 'Std':

            y = self.local_stdev(attribute+ ' Windows')

        elif metric == 'Peak to Valley':

            y = self.peak_to_valley(attribute+ ' Windows')

        return y

    def local_stdev(self,attribute):

        measureWindows = getattr(self,attribute)
        avg = measureWindows.mean()

        stdev = []

        for x in measureWindows:
            sum = 0
            for xi in x:
                sum = sum + (xi - avg) ** 2

            stdev.append(math.sqrt(sum / x.size))

        stdev = np.array(stdev)
        return stdev

    def peak_to_valley(self,attribute):

        windowedProfile = getattr(self, attribute)
        peak2valley = []

        for window in windowedProfile:
            peak = max(window)
            valley = min(window)
            peak2valley.append(peak - valley)

        self.peaktoval = peak2valley

        return peak2valley

    def trim_slopes(self,attribute,attributeX):
        X = getattr(self,attributeX)
        profile = getattr(self,attribute)
        peak = max(profile)
        peakIdx = np.where(profile == peak)[0][0]
        xtrim = X[peakIdx]

        for i in range(-1, -profile.size, -1):
            if (profile[i] > profile[i - 1]) & (profile[i] > 0.82*np.mean(profile)):
                endIdx = i
                break
            elif profile[i] < 0.82 * np.mean(profile):
                continue

        xendtrim = X[endIdx]

        profile = profile[peakIdx:endIdx]
        X = X[peakIdx:endIdx]

        setattr(self, attribute, profile)
        setattr(self, attributeX, X)

        return xtrim, xendtrim


class Bead:

    allBaseTemp = []
    alltravelSpeed = []
    allwfs = []
    allArc = []
    allctwd = []

    def __init__(self, number, lemData,weldData,robData, audio):
        self.number = number
        self.lemData = lemData
        self.weldData = weldData
        self.robData = robData
        self.audio = audio

        self.travelSpeed = None
        self.wfs = None
        self.arcCorrection = None
        self.ctwd = None
        self.arcCorrection = None
        self.meltTemps = None
        self.lineProfile = None
        self.baseTemp = None

        self.predictions = []
        self.segments = []

        self.audioFrames = {}
        self.allBeadStats = None
        self.lemDataStats = {}
        self.weldDataStats = {}
        self.meltPoolStats = {}

    def add_settings(self,settings):
        self.travelSpeed = settings['Travel']
        self.wfs = settings['WFS']

        keysList = list(settings.keys())

        if keysList[2] == 'Arc Correction':
            self.arcCorrection = settings['Arc Correction']
            Bead.allArcCorrection.append(self.arcCorrection)
        elif keysList[2] == 'Arc Correction':
            self.ctwd = settings['CTWD']
            Bead.allctwd.append(self.ctwd)

    def add_infrared(self,meltPoolVals,lineProfile):
        meltTemps = {}
        self.lineProfile = lineProfile
        self.baseTemp = np.mean(lineProfile)
        Bead.allBaseTemp.append(self.baseTemp)

        numSamples = meltPoolVals.size
        time = self.robData['Time']
        startTime = time[0]
        endTime = time[-1]
        del time
        meltTime = np.linspace(startTime,endTime,numSamples)
        meltTemps['Melt Pool Temperature'] = meltPoolVals
        meltTemps['Time'] = meltTime

        self.meltTemps = meltTemps

    def add_line_profile(self,lineProfile):
        self.lineProfile = lineProfile
        self.baseTemp = np.mean(lineProfile)
        Bead.allBaseTemp.append(self.baseTemp)

    def segment(self,overlap,numWindows):
        lemData = self.lemData
        weldData = self.weldData
        robData = self.robData

        lemData = seg.data_split_stats(lemData,overlap,numWindows)
        weldData = seg.data_split_stats(weldData, overlap, numWindows)
        robData = seg.data_split_stats(robData, overlap, numWindows)

        self.lemData = lemData
        self.weldData = weldData
        self.robData = robData

    def remove_prepost_time(self):

        robotY = self.robData['Robot Y']
        robotX = self.robData['Robot X']
        robotZ = self.robData['Robot Z']
        time = self.robData['Time']

        for i in range(0, len(robotY)):
            if robotY[i] != robotY[i + 1]:
                startIdx = i
                break
        for i in range(-1, -len(robotY), -1):
            if robotY[i] != robotY[i - 1]:
                endIdx = i
                break

        robotY = robotY[startIdx:endIdx]
        robotY = robotY - min(robotY)

        self.robData['Robot Y'] = robotY
        self.robData['Robot X'] = robotX[startIdx:endIdx]
        self.robData['Robot Z'] = robotZ[startIdx:endIdx]

        del robotX,robotY,robotZ

        startTime = time[startIdx]
        endTime = time[endIdx]
        self.robData['Time'] = time[startIdx:endIdx]

        del time

        self.weldData = trim_times(startTime, endTime, self.weldData)
        self.lemData = trim_times(startTime, endTime, self.lemData)
        self.meltTemps = trim_times(startTime, endTime, self.meltTemps)
        self.audio = trim_times(startTime, endTime, self.audio)

    def trim_profile_time(self,xtrim,xendtrim):

        robotY = self.robData['Robot Y']
        time = self.robData['Time']
        diffStart = abs(robotY - xtrim)
        idxStart= np.where(diffStart == min(diffStart))[0][0]

        diffEnd = abs(robotY - xendtrim)
        idxEnd = np.where(diffEnd == min(diffEnd))[0][0]

        startTime = time[idxStart]
        endTime = time[idxEnd]

        self.weldData = trim_times(startTime, endTime, self.weldData)
        self.lemData = trim_times(startTime, endTime, self.lemData)
        self.meltTemps = trim_times(startTime, endTime, self.meltTemps)
        self.audio = trim_times(startTime, endTime, self.audio)

    def filter_blips(self):

        xtemp = self.robData['Robot X']
        filter_idx = np.where(xtemp > 500)

        if filter_idx.size == 0:
            return 0
        else:
            self.robData['Robot X'] = np.delete(xtemp, filter_idx)
            del xtemp
            self.robData['Robot Y'] = np.delete(self.robData['Robot Y'], filter_idx)
            self.robData['Robot Z'] = np.delete(self.robData['Robot Z'], filter_idx)
            self.robData['Time'] = np.delete(self.robData['Time'], filter_idx)

            self.weldData['Wire Feed Speed'] = np.delete(self.weldData['Wire Feed Speed'], filter_idx)
            self.weldData['Time'] = np.delete(self.weldData['Time'], filter_idx)

    def delete_item(self,attribute,*itemKeys):

        attributeDict = getattr(self,attribute)
        for arg in itemKeys:
            attributeDict.pop(arg)

        setattr(self,attribute,attributeDict)

    def merge_data(self,*attributes):

        for arg in attributes:
            try:
                dataStats = getattr(self, arg)
                self.allBeadStats = self.allBeadStats | dataStats
            except:
                allBeadStats = {}
                dataStats = getattr(self, arg)
                allBeadStats = allBeadStats | dataStats
                self.allBeadStats = allBeadStats

    def segment(self,percent_overlap,num_windows):
        # Window audio data
        audio_size = self.audio['Audio'].size
        lw_audio = seg.calculate_winlength(audio_size, percent_overlap, num_windows)
        hop = floor(lw_audio - lw_audio * percent_overlap)
        # Store windowed audio features back in object
        self.audioFrames = af.extract_basic_features(self.audio['Audio'], 22050, hop, lw_audio)


        # Delete unwanted items from bead object
        self.delete_item('weldData', 'Motor Current', 'Time')
        self.delete_item('lemData', 'Time')
        self.delete_item('meltTemps', 'Time')

        # Window the rest of data and store back into object
        self.lemDataStats = seg.data_split_stats(self.lemData, percent_overlap, num_windows)
        self.weldDataStats = seg.data_split_stats(self.weldData, percent_overlap, num_windows)
        self.meltTempStats = seg.data_split_stats(self.meltTemps, percent_overlap, num_windows)

        self.merge_data('lemDataStats','weldDataStats')

        del self.audio, self.lemData, self.weldData, self.meltTempStats

    def denoise(self, group, waveform):

        wavedict = getattr(self,group)
        X_denoise = denoise_wavelet(wavedict[waveform], method='VisuShrink', mode='soft', wavelet_levels=3,
                                    wavelet='sym8', rescale_sigma='True')
        wavedict[waveform] = X_denoise
        setattr(self,group,wavedict)

class Layer:

    globalTime = None
    processActive = None

    def __init__(self,number,lemData,weldData,robData,layerTime):
        self.number = number
        self.lemData = lemData
        self.weldData = weldData
        self.robData = robData
        self.layerTime = layerTime
        self.beads = []
        self.interLayerTemp = None
        self.meltPool = None
        self.X = None
        self.Y = None

    def add_audio(self,audio,audioSR):
        self.audio = audio
        self.audioSR = audioSR

    def add_blobDetection(self,blobDetection):
        self.blobDetection = blobDetection

    def separateBeads(self):

        time = self.layerTime
        startTime = time[0]
        endTime = time[-1]

        prevBool = 1

        startIdx,endIdx = find_time_idx(self.globalTime, startTime, endTime)

        beadStartTime = []
        beadEndTime = []

        for i in range(startIdx,endIdx+1):

            currBool = self.processActive[i]

            if currBool < prevBool:
                beadEndTime.append(self.globalTime[i])
            elif currBool > prevBool:
                beadStartTime.append(self.globalTime[i])
            elif i == 1:
                beadStartTime.append(self.globalTime[i])

            prevBool = currBool

        for i in range(0,len(beadStartTime)+1):

            beadLemData = trim_times(beadStartTime[i],beadEndTime[i],self.lemData)
            beadWeldData = trim_times(beadStartTime[i],beadEndTime[i], self.weldData)
            beadRobData = trim_times(beadStartTime[i],beadEndTime[i], self.robData)

            currentBead = Bead(i,beadLemData,beadWeldData,beadRobData)
            currentBead.remove_prepost_time()

            self.beads.append(currentBead)

        del self.weldData,self.lemData,self.robData

    def compute_layer_DF(self,num_windows,percent_overlap):

        self.X, self.Y = seg.segment_assemble(self.beads, BeadShape,num_windows,percent_overlap)


def beadNumber(i):
    if i < 10:
        beadnumstr = "0" + str(i)
    else:
        beadnumstr = str(i)

    return beadnumstr


def get_LEMtime(data):
    time = []
    sample_rate = 20000
    n_samples = len(data['Welding Voltage'])
    dt = 1/sample_rate

    for i in range(1,n_samples+1):
        time.append(dt*i)

    return time


def extract_tdms(path,NumPts):

    beads = []

    for i in range(1, NumPts+1):

        TempRobData = {}
        TempWeldData = {}
        TempLEMData = {}

        beadnumstr = beadNumber(i)

        temp_file = TdmsFile.read(path + '\\LabVIEW\\Bead' + beadnumstr + '\\Bead01.tdms')

        RobGroup = temp_file["Robot Data"]
        WeldGroup = temp_file["Welding Data"]
        LEMGroup = temp_file['LEM Box']

        for channel in RobGroup.channels():
            TempRobData[channel.name] = channel[:]

        for channel in LEMGroup.channels():
            TempLEMData[channel.name] = channel[:]

        for channel in WeldGroup.channels():
            TempWeldData[channel.name] = channel[:]

        LEMtime = get_LEMtime(TempLEMData)
        TempLEMData['Time'] = LEMtime

        beads.append(Bead(i,TempLEMData,TempWeldData,TempRobData))

    return beads


def extract_IR_data(path,beads,num):
    try:
        pathBlob = path + '\\IR Data\\Blob Detection\\'
        pathLine = path + '\\IR Data\\Line Profiles\\'
        for i in range(1, num + 1):
            beadnumstr = beadNumber(i)

            filenameLine = 'Bead' + beadnumstr + '_basetemp'
            filenameBlob = 'Bead' + beadnumstr + '_BlobDetection'

            temp_dfLine = pd.read_csv(pathLine + filenameLine + '.csv')
            temp_dfBlob = pd.read_csv(pathBlob + filenameBlob + '.csv')

            blobTemps = temp_dfBlob['TemperatureMean'].values
            lineTemps = temp_dfLine['Bead' + beadnumstr + '.seq:Line 1 [C]:mean:vert'].values

            beads[i - 1].add_infrared(blobTemps, lineTemps)
    except:
        pathLine = path + '\\IR Data\\Line Profiles\\'
        for i in range(1, num + 1):
            beadnumstr = beadNumber(i)
            filenameLine = 'Bead' + beadnumstr + '_basetemp'
            temp_dfLine = pd.read_csv(pathLine + filenameLine + '.csv')
            lineTemps = temp_dfLine['Bead' + beadnumstr + '.seq:Line 1 [C]:mean:vert'].values
            beads[i - 1].add_line_profile(lineTemps)

    return beads


def extract_tdms_layers(path,NumPts):

    Layers = []

    tc_file = TdmsFile.read(path +'\\LabVIEW\\Build_220705_h07.tdms')
    processActive = tc_file["Process Active"]
    allLayerTimes = tc_file['Layer Times']

    Layer.globalTime = processActive['Time']
    Layer.processActive = processActive["Bool"][:]

    for i in range(1, NumPts):

        tempRobData = {}
        tempWeldData = {}
        tempLEMData = {}

        beadnumstr = beadNumber(i)

        temp_file = TdmsFile.read(path +'\\LabVIEW\\Layer'+beadnumstr+'.tdms')

        RobGroup = temp_file["Robot Data"]
        WeldGroup = temp_file["Welding Data"]
        LEMGroup = temp_file["LEM Box"]

        for channel in RobGroup.channels():
            tempRobData[channel.name] = channel[:]

        for channel in WeldGroup.channels():
            tempWeldData[channel.name] = channel[:]

        for channel in LEMGroup.channels():
            tempLEMData[channel.name] = channel[:]

        LEMtime = get_LEMtime(tempLEMData)
        tempLEMData['LEM Time'] = LEMtime
        layerTime = allLayerTimes['Layer'+ beadnumstr]

        Layers.append(Layer(i,tempLEMData,tempWeldData,tempRobData,layerTime))

    return Layers


# def standardize_data(X):
#
#     for (columnName, columnData) in X.iteritems():
#         featureVector = columnData.values
#         featureAvg = featureVector.mean()
#         featureStd = featureVector.std()
#         normFeature = (featureVector - featureAvg)/featureStd
#         X[columnName] = normFeature
#
#     return X


def normalize_data(X):

    for (columnName, columnData) in X.iteritems():
        featureVector = columnData.values
        featureMax = featureVector.max()
        featureMin = featureVector.min()
        normFeature = (featureVector - featureMin)/(featureMax - featureMin)
        X[columnName] = normFeature

    return X


def find_time_idx(time, startTime,endTime):

    diff = abs(time - startTime)
    startIdx = np.where(diff == min(diff))[0][0]

    diffEnd = abs(time - endTime)
    endIdx = np.where(diffEnd == min(diffEnd))[0][0]

    return startIdx,endIdx


def trim_times(startTime,endTime,data):

    time = data['Time']
    startIdx, endIdx = find_time_idx(time,startTime,endTime)

    del time

    for key in data:
        series = data[key]
        data[key] = series[startIdx:endIdx]

    return data


def extract_labview(path,NumPts):

    beads = []

    for i in range(1, NumPts+1):

        TempRobData = {}
        TempWeldData = {}
        TempLEMData = {}
        Audio = {}

        beadnumstr = beadNumber(i)
        beadPath = path+'\\LabVIEW\\Bead' + beadnumstr

        temp_file = TdmsFile.read(beadPath + '\\Bead01.tdms')

        RobGroup = temp_file["Robot Data"]
        WeldGroup = temp_file["Welding Data"]
        LEMGroup = temp_file['LEM Box']

        for channel in RobGroup.channels():
            TempRobData[channel.name] = channel[:]

        for channel in LEMGroup.channels():
            TempLEMData[channel.name] = channel[:]

        for channel in WeldGroup.channels():
            TempWeldData[channel.name] = channel[:]

        LEMtime = get_LEMtime(TempLEMData)
        TempLEMData['Time'] = LEMtime

        finalTime = TempRobData['Time'][-1]

        #Get Audio
        tempAudio, SR = none = librosa.load(beadPath + '\\Bead01.wav')
        tempAudio = af.clipEndAudio(tempAudio)
        time = af.add_time(tempAudio,finalTime)
        Audio['Time'] = time
        del time
        Audio['Audio'] = tempAudio
        beads.append(Bead(i, TempLEMData, TempWeldData, TempRobData, Audio))
        del tempAudio

    return beads


