import librosa
import matplotlib
import os
import numpy as np
from math import floor

import librosa.display
import matplotlib.pyplot as plt,scipy
import pandas as pd

from skimage.restoration import denoise_wavelet


def extract_basic_features(audio, sampling_rate = 48000, hop = 512, n_win = 2048) :

    spectral_centroids = librosa.feature.spectral_centroid((audio-np.mean(audio)), sr=sampling_rate, n_fft=n_win, hop_length=hop, center = False)[0]
    spectral_centroids = spectral_centroids[:-1]
    spectral_rolloff = librosa.feature.spectral_rolloff((audio-np.mean(audio)), sr=sampling_rate, n_fft=n_win, hop_length=hop, center = False)[0]
    spectral_rolloff = spectral_rolloff[:-1]
    spectral_bandwidth = librosa.feature.spectral_bandwidth((audio-np.mean(audio)), sr=sampling_rate, n_fft=n_win, hop_length=hop, center = False)[0]
    spectral_bandwidth = spectral_bandwidth[:-1]
    rms = librosa.feature.rms(audio, frame_length =n_win, hop_length=hop, center = False)[0]
    rms = rms[:-1]
    zero_crossing = librosa.feature.zero_crossing_rate(audio, frame_length =n_win, hop_length=hop, center = False)[0]
    zero_crossing = zero_crossing[:-1]

    tempo = librosa.beat.tempo(audio, sr=sampling_rate, hop_length = hop)[0]

    df = pd.DataFrame()

    df['Spectral Centroids'] = spectral_centroids
    df["Spectral Bandwidth"] = spectral_bandwidth
    df['Spectral Rolloff'] = spectral_rolloff
    df['Root Mean Squared'] = rms
    df['Zero Crossing Rate'] = zero_crossing
    df['Tempo'] = tempo

    #harmonics = librosa.effects.hpss(analysis_samples)
    #beat_track  = librosa.beat.beat_track(analysis_samples)


    #df['Harmonics'] = harmonics
    #df['Perpetual_Shock'] = perpetual_shock

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








