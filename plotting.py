import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import floor
import os
from scipy import signal

import audio_features as af

from tkinter import *
from tkinter import filedialog


def plot_main():
    plotMenu = Tk()
    plotMenu.title('Plotting Menu')


def plot_histogram(Y,metric):

    hist = plt.figure()

    Y = np.array(Y)

    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=Y, bins=30, color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(metric)
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    hist.savefig(metric+' Histogram.png')

    return n,bins


def plot_profile(beadshape, profileType,path):

    try:
        output_path = os.path.join(path,profileType)
        os.mkdir(output_path)
    except:
        p = None

    profilePlot = plt.figure()

    z = beadshape.Height
    x = beadshape.x
    num = beadshape.beadNum

    plt.plot(x,z)
    plt.xlabel('Length (mm)')
    plt.ylabel('Height (mm)')

    profilePlot.savefig(output_path+'\\' + profileType+ ' Bead'+str(num)+'.png')

    plt.close()


def plot_output(beadshape,path,metric):

    try:
        output_path = os.path.join(path, 'Output Profiles')
        os.mkdir(output_path)
    except:
        p = None

    profilePlot = plt.figure()

    peaktoval = beadshape.peaktoval
    num = beadshape.beadNum
    overlap = beadshape.overlap
    num_windows = beadshape.numWindows

    x = beadshape.segment(overlap,num_windows,attribute = 'x',metric = 'Mean')

    plt.plot(x,peaktoval)
    plt.xlabel('Length (mm)')
    plt.ylabel('Height (mm)')

    profilePlot.savefig(output_path+'\\'+ metric+ ' Profile Bead ' + str(num) +'png')


def plot_segmentXY(X,Y,input,units):
    fig = plt.figure()
    data = X[input]; plt.scatter(data,Y)
    print(units[input])
    plt.xlabel(input + ' (' + units[input][0] + ')')
    plt.ylabel('Height Average (mm)')
    fig.savefig('XY Plot' + input + '.png')

def plot_timeseries(WeldData,key):

    time = WeldData[key]['Time']
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('Voltage (V)')
    ax1.plot(time, WeldData[key]['Welding Voltage'], color="blue", label='Voltage')


    ax2 = ax1.twinx()

    ax2.plot(time, WeldData[key]['Welding Current'], color="green", label='Current')
    ax2.set_ylabel('Current (A)')

    # ax3 = ax1.twinx()
    #
    # ax3.plot(time, WeldData[key]['Wire Feed Speed'], color="red", label='Wire Feed Speed')
    # ax3.spines['right'].set_position(('outward', 60))
    # ax3.set_ylabel('Wire Feed Speed (in/min)')

    fig1.legend()
    plt.show()
    # fig1.savefig('Time Series\\Time_Series_' + key + '.png')



def all_build_correlations(dict, units, time_series1, time_series2):

    plt.figure()
    for key in dict:
        plt.scatter(dict[key][time_series1], dict[key][time_series2], color='none', edgecolor='blue')
        plt.xlabel(time_series1 + units[time_series1])
        plt.ylabel(time_series2 + units[time_series2])

    plt.show()


def plot_autocorr(Data, bead, *data_stream):

    for arg in data_stream:
        plt.figure()
        plt.acorr(Data[bead][arg],maxlags=100)
        plt.show()
        plt.xlabel('Lags')
        plt.ylabel(arg+' Autocorrelation')


def plot_xy(Data, units, layer, time_series1, time_series2):
    plt.figure()
    plt.scatter(Data['Bead'+layer][time_series1], Data['Bead'+layer][time_series2])
    plt.xlabel(time_series1 + units[time_series1])
    plt.ylabel(time_series2 + units[time_series2])


def plot_bead_data(WeldData, Audio, audio_sample_rate, bead):

    time_series = ['Welding Voltage', 'Welding Current', 'Wire Feed Speed', 'Travel Speed']

    plot_timeseries(WeldData, bead)
    plot_autocorr(WeldData, bead, *time_series)
    af.plot_audio_spectra(Audio, audio_sample_rate, bead)

    for i in range(0, 3):

        for j in range(1, 4):

            if i is not j:
                plot_xy(WeldData, bead, time_series[i],time_series[j])


def plot_data_spectra(Data, bead, time_series, cut_freq):

    arc_plot = plt.figure()
    x = Data[bead][time_series]

    fourier_trans = np.fft.fft(x - np.mean(x))
    mag_spectrum = np.abs(fourier_trans)

    N = len(mag_spectrum)

    sr = 20000

    frequency = np.linspace(0, (sr / 2)+1, floor(N/2))
    cut_idx = np.where(frequency > cut_freq)

    plt.plot(frequency[:cut_idx[0][0]], mag_spectrum[:cut_idx[0][0]])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    power_plot = plt.figure()

    fourier_conj = np.conjugate(fourier_trans)

    # Sxx = (2/N**2)*(fourier_conj*fourier_trans)
    #
    # plt.plot(frequency[1:], Sxx[1:floor(N/2)])
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power')

    return arc_plot, frequency, mag_spectrum[0:floor(N/2)]


def parameter_correlations(X,Y, Xunit, Yunit):

    for ycol in Y.columns.to_numpy():
        i = 0
        for xcol in X.columns.to_numpy():

            if i < 8:
                plt.figure()
                plt.scatter(X[xcol], Y[ycol])
                plt.ylabel(ycol + Yunit[ycol], fontsize=20)
                plt.xlabel(xcol + '(' + Xunit[xcol].item() + ')', fontsize=20)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)


                plt.show()

                i = i + 1


def spectra_allbeads(welddata, audio, cut_freq, NumPts):
    plt.ioff()

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
        volt_plot, volt_freq, volt_mag = plot_data_spectra(welddata, Bead, 'Welding Voltage', cut_freq)
        curr_plot, curr_freq, curr_mag = plot_data_spectra(welddata, Bead, 'Welding Current', cut_freq)
        audio_plot, audio_freq, audio_mag = af.plot_audio_spectra(audio, 22050, Bead, cut_freq)

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


def color_correlation(X,Y, settings, Xunit, Yunit):

    colors = ['blue','green','red','yellow']
    markers = ['^', 'o', '*', 's']

    wfs = settings['WFS']
    ts = settings['Travel Speed']
    arc = settings['Arc Correction']

    for i in range(0,5):
        wfs[wfs == 100 + i*50] = i
        ts[ts == 6 + i * 2] = i

    # arc[arc == -30] = 0
    # arc[arc == -8] = 1
    # arc[arc == 15] = 2

    wfs_levels = ['100' , '150' , '200', '250']
    ts_levels = ['6' , '8', '10' , '12']
    arc_levels = ['-30', '-8', '15']
    X = X.drop(['Wire Feed Speed Mean', 'Wire Feed Speed Std','Travel Speed Mean'], axis = 1)

    for ycol in Y.columns.to_numpy():

        i = 0
        for xcol in X.columns.to_numpy():

            n = 0

            if i < 5:
                plt.figure()

                for n in range(0,len(wfs)):

                    plt.scatter(X[xcol][n], Y[ycol][n] , marker= markers[ts[n]], color = colors[wfs[n]])
                    plt.ylabel(ycol + Yunit[ycol], fontsize=20)
                    plt.xlabel(xcol + '(' + Xunit[xcol].item() + ')', fontsize=20)
                    plt.xticks(fontsize=16)
                    plt.yticks(fontsize=16)

                i = i + 1

                f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

                handles = [f("s", colors[j]) for j in range(4)]
                handles += [f(markers[j], "k") for j in range(4)]

                labels = wfs_levels + ts_levels

                plt.legend(handles, labels, bbox_to_anchor = (1.12,1), loc="upper right", framealpha=1)

                plt.show()
