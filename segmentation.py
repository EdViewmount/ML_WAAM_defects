import numpy as np
import pandas as pd
from math import floor

import librosa

import unittest
from numpy.testing import assert_array_almost_equal,             assert_almost_equal, assert_equal
import warnings

class Segment:
    def __init__(self, data, bead_num, seg_num):
        self.data = data
        self.predictions = 0
        self.bead_num = bead_num
        self.seg_num = seg_num


def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    if axis is None:
        a = np.ravel(a)  # may copy
        axis = 0

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
    lw = floor((L + num_windows * overlap - overlap) / num_windows)

    return lw


def calculate_winlength_audio(L, overlap, num_windows):
    lw = floor((L + num_windows * overlap) / (num_windows + 1))

    return lw


def data_split(dict, lw, o):
    stats = {}

    for key in dict:
        temp = dict[key]
        temp = segment_axis(temp, lw, overlap=overlap, end="cut")

        stats[key + ' Mean'] = np.mean(temp, axis=1)
        stats[key + ' Std'] = np.std(temp, axis=1)

    return stats


def store_segment_data(segmented_beads, AudioFrames, num_windows):
    Segments = []
    seg_num = 0

    for key in segmented_beads:
        bead_num = int(key[-1])
        bead_data = segmented_beads[key]
        audio_data = AudioFrames[key]
        for i in range(0, num_windows):
            segment_data = pd.DataFrame()
            for key in bead_data:
                temp = (bead_data[key][i])
                segment_data[key] = [temp]
            temp_audio = audio_data.iloc[[i]]
            temp_audio.reset_index(drop=True, inplace=True)
            segment_data = pd.concat([segment_data, temp_audio], axis=1)
            Segments.append(Segment(segment_data, bead_num, seg_num))
            seg_num += 1


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

def segment_data(LEMData,WeldData,num_windows):

    return 1
