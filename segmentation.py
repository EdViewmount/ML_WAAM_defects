import numpy as np
import pandas as pd
from math import floor, ceil
import scipy as sc

import unittest
from numpy.testing import assert_array_almost_equal,             assert_almost_equal, assert_equal

import audio_features as af
import data_processing as dp
import warnings


class Segment:

    X = pd.DataFrame()

    def __init__(self, data, bead_num, globalSegNum):
        self.data = data
        self.predictions = 0
        self.bead_num = bead_num
        self.globalSegNum = globalSegNum
        self.segNum = None

    def assign_predictions(self,predictions):
        idx = self.seg_num
        seg_pred = predictions[idx]
        self.predictions = seg_pred


def segment_axis(a, length, percent_overlap=0, axis=None, end='cut', endvalue=0):
    """Function takes in 1D array of data and converts it into frames of overlapping windows.
    The 1D array is converted to an ndarray where the rows correspond to one window of data. The overlap
    is expressed in terms of percent overlap (percent of window length)"""

    if axis is None:
        a = np.ravel(a)  # may copy
        axis = 0

    overlap = ceil(length * percent_overlap)

    l = a.shape[axis]

    if overlap >= length:
        raise ValueError("frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0:
        raise ValueError("overlap must be nonnegative and length must be positive")

    if l < length or (l - length) % (length - overlap):
        if l > length:
            roundup = length + (1 + (l - length) // (length - overlap)) * (length - overlap)
            rounddown = length + ((l - length) // (length - overlap)) * (length - overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length - overlap) or (roundup == length and rounddown == 0)
        a = a.swapaxes(-1, axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad', 'wrap']:  # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s, dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup - l]
            a = b

        a = a.swapaxes(-1, axis)

    l = a.shape[axis]
    if l == 0:
        raise ValueError("Not enough data points to segment array in 'cut' mode; try 'pad' or 'wrap'")
    assert l >= length
    assert (l - length) % (length - overlap) == 0
    n = 1 + (l - length) // (length - overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n, length) + a.shape[axis + 1:]
    newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides, shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides, shape=newshape, buffer=a, dtype=a.dtype)


def calculate_winlength(L, overlap, num_windows):
    lw = floor(L/(num_windows - overlap*num_windows + overlap))

    return lw


def data_split_stats(dict,overlap,num_windows):
    """Splits all the 1d arrays of a data dictionary into an ndarray and calculates stats for each segment"""
    stats = {}

    for key in dict:
        data = dict[key]
        N = len(data)
        lw = calculate_winlength(N, overlap, num_windows)
        data = segment_axis(data, lw, percent_overlap = overlap, end="cut")

        stats[key + ' Mean'] = np.mean(data, axis=1)
        stats[key + ' Std'] = np.std(data, axis=1)
        stats[key + ' Kurt'] = sc.stats.kurtosis(data,axis=1)

    return stats


def assemble_df(Segments):
    i = 1
    for segment in Segments:

        xtemp = segment.data
        if i == 1:
            X = xtemp
        else:
            X = pd.concat([X, xtemp], axis=0, ignore_index=True)
        i += 1
    return X


def output_array(BeadShapes,attribute,metric, overlap,num_windows):

    Y = []
    for beadshape in BeadShapes:
        Y.extend(beadshape.segment(overlap,num_windows,attribute = attribute,metric = metric))

    return Y


def store_segment_data(Beads, num_windows):
    Segments = []
    globalNum = 0

    for bead in Beads:
        bead_num = bead.number
        bead_data = bead.allBeadStats
        audio_data = bead.audioFrames
        travelSpeed = bead.travelSpeed
        baseTemp = bead.baseTemp

        segNum = 0

        for i in range(0, num_windows):
            segment_data = pd.DataFrame()
            for key in bead_data:
                temp = (bead_data[key][i])
                segment_data[key] = [temp]
            temp_audio = audio_data.iloc[[i]]
            temp_audio.reset_index(drop=True, inplace=True)
            segment_data = pd.concat([segment_data, temp_audio], axis=1)
            segment_data['Travel Speed'] = travelSpeed
            segment_data['Base Temperature'] = baseTemp
            tempSegment = Segment(segment_data, bead_num, globalNum)
            Segments.append(tempSegment)
            globalNum += 1



    return Segments


def segment_assemble(Beads, Output, num_windows,percent_overlap,attribute = 'z', metric = 'Std'):

    for bead in Beads:
        bead.segment(percent_overlap,num_windows)

    Segments = store_segment_data(Beads, num_windows)

    X = assemble_df(Segments)
    Y = output_array(Output,attribute,metric,percent_overlap,num_windows)

    return X, Y

def layer_assemble(Layers):

    X = pd.DataFrame()

    for layer in Layers:
        x = layer.X
        y = layer.Y


def prediction_assignment(Segments, Beads, predictions):
    for segment in Segments:
        segment.assign_predictions(predictions)

    for segment in Segments:
        bead_curr = segment.bead_num
        Beads[bead_curr].predictions.append(segment.predictions)


# def prediction_assignment(Segments,predictions):
#     for segment in Segments:
#         segment.assign_predictions(predictions)
#
#     Bead_preds = {}
#     i = 1
#     bead_prev = 0
#     for segment in Segments:
#         bead_curr = segment.bead_num
#         if bead_curr > bead_prev: i = 1
#
#         if i == 1:
#             Bead_preds['Bead' + str(bead_curr)] = []
#             Bead_preds['Bead' + str(bead_curr)].append(segment.predictions)
#         else:
#             Bead_preds['Bead' + str(bead_curr)].append(segment.predictions)
#
#         i += 1
#         bead_prev = bead_curr

