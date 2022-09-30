import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

    # time = WeldData['Bead' + layer_num]['Time']
    # fig2, ax1 = plt.subplots()
    # ax1.set_xlabel('Time(s)')
    # ax1.set_ylabel('Travel Speed (in/min')
    # ax1.plot(time, WeldData['Bead' + layer_num]['Travel Speed'], color="blue")
    #
    # plt.show()


def all_build_correlations(dict, units, time_series1, time_series2):

    plt.figure()
    for key in dict:
        plt.scatter(dict[key][time_series1], dict[key][time_series2], color='none', edgecolor='blue')
        plt.xlabel(time_series1 + units[time_series1])
        plt.ylabel(time_series2 + units[time_series2])

    plt.show()


def plot_autocorr(Data, layer, *data_stream):

    for arg in data_stream:
        plt.figure()
        plt.acorr(Data['Bead'+layer][arg],maxlags=100)
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


def plot_data_spectra(Data, layer, time_series):

    plt.figure()

    fourier_trans = np.fft.fft(Data['Bead'+layer][time_series])
    mag_spectrum = np.abs(fourier_trans)

    N = len(mag_spectrum)

    sr = 2.77

    frequency = np.linspace(0, (sr / 2)+1, N)

    plt.plot(frequency, mag_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.figure()

    fourier_conj = np.conjugate(fourier_trans)

    Sxx = (2/N^2)*(fourier_conj*fourier_trans)

    plt.plot(frequency, Sxx)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')


def parameter_correlations(X,Y, Xunit, Yunit):

    Y = Y.drop('Porosity',axis=1)

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


def color_correlation(X,Y, Xunit, Yunit):

    colors = ['b' ,'g','r','c']

    idx = []
    wfs= X['Wire Feed Speed Mean']
    wfs_levels = ['100' , '150' , '200', '250']

    for i in range(0,wfs.size):

        if i == 0: templist = []

        if i > 0 and wfs[i] - wfs[i-1] > 10:
            idx.append(templist)
            templist = []
        elif i == wfs.size - 1:
            templist.append(i)
            idx.append(templist)
        else:
            templist.append(i)

    print(idx)

    for ycol in Y.columns.to_numpy():

        i = 0
        for xcol in X.columns.to_numpy():

            n = 0

            if i < 7:
                ax = plt.figure()

                for n in range(0,len(idx)):

                    plt.scatter(X[xcol][idx[n]], Y[ycol][idx[n]] , c = colors[n], label = wfs_levels[n])
                    plt.ylabel(ycol + Yunit[ycol], fontsize=20)
                    plt.xlabel(xcol + '(' + Xunit[xcol].item() + ')', fontsize=20)
                    plt.xticks(fontsize=16)
                    plt.yticks(fontsize=16)

                    plt.show()

                i = i + 1
                ax.legend(title = 'WFS')



