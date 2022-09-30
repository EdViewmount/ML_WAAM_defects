import librosa
import matplotlib
import os
import numpy as np
from math import floor

import soundfile
import librosa.display
import matplotlib.pyplot as plt,scipy
import IPython.display as ipd
import pandas as pd

import pywt
from skimage.restoration import denoise_wavelet


def extract_basic_features(analysis_samples, sampling_rate) :

    spectral_centroids = {}
    spectral_rolloff = {}
    spectral_bandwidth = {}
    rms = {}
    zero_crossing={}
    tempo = {}
    #chromagram = {}
    #chroma_01 = {}
    #chroma_02 = {}
    #chroma_03 = {}
    #chroma_04 = {}
    #chroma_05 = {}
    #chroma_06 = {}
    #chroma_07 = {}
    #chroma_08 = {}
    #chroma_09 = {}
    #chroma_10 = {}
    #chroma_11 = {}
    #chroma_12 = {}


    spectral_centroids = librosa.feature.spectral_centroid(analysis_samples, sr=sampling_rate)[0]
    spectral_centroids = spectral_centroids[:-1]
    #spectral_centroids_delta = librosa.feature.delta(spectral_centroids)
    #spectral_centroids_accelerate = librosa.feature.delta(spectral_centroids, order=2)
    spectral_rolloff = librosa.feature.spectral_rolloff(analysis_samples+0.01, sr=sampling_rate)[0]
    spectral_rolloff = spectral_rolloff[:-1]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(analysis_samples+0.01, sr=sampling_rate)[0]
    spectral_bandwidth = spectral_bandwidth[:-1]
    rms = librosa.feature.rms(analysis_samples+0.01)[0]
    rms = rms[:-1]
    zero_crossing = librosa.feature.zero_crossing_rate(analysis_samples)[0]
    zero_crossing = zero_crossing[:-1]

    tempo = librosa.beat.tempo(analysis_samples, sr=sampling_rate)[0]





    df = pd.DataFrame()

    df['Spectral Centroids'] = spectral_centroids
    #df['Spectral_Centroids_Delta'] = spectral_centroids_delta
    #df['Spectral_Centroids_Accelerate'] = spectral_centroids_accelerate
    df["Spectral Bandwidth"] = spectral_bandwidth
    df['Spectral Rolloff'] = spectral_rolloff
    df['Root Mean Squared'] = rms
    df['Zero Crossing Rate'] = zero_crossing
    df['Tempo'] = tempo

    #harmonics = librosa.effects.hpss(analysis_samples)
    #beat_track  = librosa.beat.beat_track(analysis_samples)


    #df['Harmonics'] = harmonics
    #df['Perpetual_Shock'] = perpetual_shock
    #df['Beat_Track'] = beat_track

    return df

def trim_audio(x):

    Endpoints=[]
    for i in range(1, len(x)):
        if x[i] > 0.01:
            Endpoints.append(i)

    Endpoints=np.array(Endpoints)
    start=Endpoints[1]
    end=Endpoints[len(Endpoints)-1]
    x=x[start:end]

    return x


def fourier_transform(X):
    Xfft=scipy.fftpack.fft(X)
    X_mag=np.absolute(X)

    return Xfft, X_mag

def file_extract(NumPts,pathAudio):
    Audio={}
    SR={}
    for i in range(1,NumPts+1):

        if i < 10:
            beadnumstr = "0" + str(i)
        else:
            beadnumstr = str(i)

        key = "Bead"+str(i)

        path = pathAudio+beadnumstr
        files=librosa.util.find_files(path, ext='wav')
        Audio[key],SR[key]=none=librosa.load(files[0])

    return Audio, SR

def denoise_audio(Audio):
    for key in Audio:
        X = Audio[key]
        X_denoise = denoise_wavelet(X, method='VisuShrink',mode='soft',wavelet_levels=3,wavelet='sym8',rescale_sigma='True')
        Audio[key]=X_denoise


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

def plot_audio_spectra(Audio, sr, layer):

    plt.figure()

    fourier_trans = np.fft.fft(Audio['Bead'+layer])
    mag_spectrum = np.abs(fourier_trans)

    N = len(mag_spectrum)


    frequency = np.linspace(0, (sr / 2)+1,floor(N/2) )

    plt.plot(frequency, mag_spectrum[0:floor(N/2)])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.figure()

    fourier_conj = np.conjugate(fourier_trans)

    Sxx = (2/N**2)*(fourier_conj*fourier_trans)

    plt.plot(frequency, Sxx[0:floor(N/2)])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')







