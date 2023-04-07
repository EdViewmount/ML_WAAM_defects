import math

import librosa
import scipy as sc
import numpy as np
from math import floor
import librosa.display
import matplotlib.pyplot as plt,scipy
import pandas as pd

from skimage.restoration import denoise_wavelet


def extract_basic_features(audio, sampling_rate = 48000, hop = 512, n_win = 2048):

    spectral_centroids = librosa.feature.spectral_centroid((audio-np.mean(audio)), sr=sampling_rate, n_fft=n_win, hop_length=hop, center = False)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff((audio-np.mean(audio)), sr=sampling_rate, n_fft=n_win, hop_length=hop, center = False)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth((audio-np.mean(audio)), sr=sampling_rate, n_fft=n_win, hop_length=hop, center = False)[0]
    rms = librosa.feature.rms(audio, frame_length =n_win, hop_length=hop, center = False)[0]
    zero_crossing = librosa.feature.zero_crossing_rate(audio, frame_length =n_win, hop_length=hop, center = False)[0]

    spec = np.abs(librosa.stft(audio, n_fft=n_win, hop_length=hop, center=False))
    spec = librosa.amplitude_to_db(spec, ref=np.max)

    onset = librosa.onset.onset_strength(audio, sr=sampling_rate, S=spec, hop_length=hop)

    del spec, audio

    df = pd.DataFrame()

    df['Spectral Centroids'] = spectral_centroids
    df["Spectral Bandwidth"] = spectral_bandwidth
    df['Spectral Rolloff'] = spectral_rolloff
    df['Root Mean Squared'] = rms
    df['Zero Crossing Rate'] = zero_crossing
    df['Onset Strength'] = onset

    return df


def extract_timeseries_features(audio,lw_audio,hop):

    audioWindows = librosa.util.frame(audio,lw_audio,hop)

    audioAvgs = np.mean(audioWindows, axis=0)
    audioKurt = sc.stats.kurtosis(audioWindows, axis=0)
    soundPressureLevel = sound_pressure(audio,150)
    rms = librosa.feature.rms(audio, frame_length=lw_audio, hop_length=hop, center=False)[0]

    audioFeat = pd.DataFrame()

    audioFeat['Amplitude Mean'] = audioAvgs
    audioFeat['Amplitude Kurtosis'] = audioKurt
    audioFeat['Sound Pressure Level'] = soundPressureLevel
    audioFeat['Root Mean Square'] = rms

    return audioFeat

def add_time(audio, endTime):

    numSamples = audio.size
    time = np.linspace(0,endTime,numSamples)

    return time


def sound_pressure(audioSample,sensitivity):
    Pref = 20*10**-6 #Pa

    soundPressureLevel = []

    for row in audioSample:

        soundPressure = audioSample/sensitivity
        soundPressureLevel.append(20*math.log10(soundPressure.mean()/Pref))

    return soundPressureLevel


def clipEndAudio(audioex):
    for i in range(-1, -len(audioex), -1):
        if audioex[i] > 0.7:
            endidx = i
            break

    audioex = audioex[0:endidx]

    return audioex


def fourier_transform(X):
    Xfft=scipy.fftpack.fft(X)
    X_mag=np.absolute(X)

    return Xfft, X_mag


def file_extract(NumPts,pathAudio,Beads):

    for i in range(1,NumPts+1):

        Audio = {}

        if i < 10:
            beadnumstr = "0" + str(i)
        else:
            beadnumstr = str(i)

        path = pathAudio+'\\LabVIEW\\Bead' + beadnumstr

        #Find all audio files in target directory
        files=librosa.util.find_files(path, ext='wav')
        tempAudio,SR=none=librosa.load(files[0])

        # Remove the portion of the audio that recorded after the arc shut off
        tempAudio = clipEndAudio(tempAudio)

        finalTime = Beads[i-1].robData['Time'][-1]
        time = add_time(tempAudio,finalTime)
        Audio['Time'] = time
        del time
        Audio['Audio'] = tempAudio
        Beads[i-1].add_audio(Audio,SR)
        del tempAudio

    return Beads


def denoise_audio(Beads):

    for bead in Beads:
        X_denoise = denoise_wavelet(bead.audio['Audio'], method='VisuShrink',mode='soft',wavelet_levels=3,wavelet='sym8',rescale_sigma='True')
        bead.audio['Audio']=X_denoise
    return Beads


def file_extract_layers(pathAudio):
    Layers = []

    files=librosa.util.find_files(pathAudio, ext='wav')

    i=1
    for filename in files:
        key='Layer'+str(i)
        sr='SR'+str(i)
        tempAudio,SR=none=librosa.load(filename)
        i=i+1
        # Remove the portion of the audio that recorded after the arc shut off
        tempAudio = clipEndAudio(tempAudio)
        Layers[i - 1].add_audio(tempAudio, SR)

    return Layers


def feature_averages(X):
    size=X.shape
    col=size[0]
    row=size[1]
    ColNames=list(X.columns)
    d={}

    for i in range(0,row):
        dftemp4=X.iloc[0:col,i].mean()
        d[ColNames[i]+' Mean']=[dftemp4]
        dftemp5=X.iloc[0:col,i].skew()
        d[ColNames[i]+' Skew']=[dftemp5]
        dftemp6=X.iloc[0:col,i].std()
        d[ColNames[i]+' Std']=[dftemp6]
        dftemp8=X.iloc[0:col,i].kurt()
        d[ColNames[i]+' Kurt']=[dftemp8]

    return d


def plot_audio_spectra(Audio, sr, layer, cut_freq):

    audio_plot = plt.figure()
    x = Audio[layer]

    fourier_trans = np.fft.fft(x - np.mean(x))
    mag_spectrum = np.abs(fourier_trans)

    N = len(mag_spectrum)

    frequency = np.linspace(0, (sr / 2)+1,floor(N/2))
    cut_idx = np.where(frequency > cut_freq)

    plt.plot(frequency[0:cut_idx[0][0]], mag_spectrum[0:cut_idx[0][0]])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    power_plot = plt.figure()

    fourier_conj = np.conjugate(fourier_trans)

    # Sxx = (2/N**2)*(fourier_conj*fourier_trans)
    #
    # plt.plot(frequency, Sxx[0:floor(N/2)])
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power')

    return audio_plot, frequency, mag_spectrum[0:floor(N/2)]





