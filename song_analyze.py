# coding: utf-8
import librosa
import numpy as np
import os
import pandas as pd
import scipy.signal

import cqt
import melody_extraction as meloext

def locmax(vec, indices=False):
    """
    Return a boolean vector of which points in vec are local maxima.
    End points are peaks if larger than single neighbors.
    if indices=True, return the indices of the True values instead of the boolean vector.
    """
    nbr = np.zeros(len(vec)+1, dtype=bool)
    nbr[0] = True
    nbr[1:-1] = np.greater_equal(vec[1:], vec[-1])
    maxmask = (nbr[:-1] & ~nbr[1:])

    if indices:
        return np.nonzero(maxmask)[0]
    else:
        return maxmask

# DENSITY controls the density of landmarks found (approx DENSITY per sec)
DENSITY = 20.0
# OVERSAMP > 1 tries to generate extra landmarks by decaying faster
OVERSAMP = 1
N_FFT = 512
N_HOP = 256
# spectrogram enhancement
HPF_POLE = 0.98

class Song:
    # optimaization: cache pre-calculated Gaussian profile
    __sp_width = None
    __sp_len = None
    __sp_vals = []

    def __init__(self, path, feature, is_extract, is_decompose):
        self.density = DENSITY
        self.n_fft = N_FFT
        self.n_hop = N_HOP
        #how wide to spread peaks
        self.f_sd = 30.0
        # Maximum number of local maxima to keep per frame
        self.maxpksperframe = 50

        self.path = path
        self.feature = feature
        self.filename = (os.path.splitext(path)[0]).split("/")[-1]
        self.h, self.g = self._path2g(feature, is_extract, is_decompose)
        self.htr = self.h

    def _path2g(self, feature, is_extract, is_decompose):
        print ("### " + self.filename + " load...")

        # extract chroma
        y, sr = librosa.load(self.path)

        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, start_bpm=180)
        print(tempo)

        y_harmonic, y_percussive = librosa.effects.hpss(y)
        chroma_cqt = cqt.chroma_cqt(y=y_harmonic, sr=sr, tempo=tempo, lwintype='note', threshold=0.0)

        ha_sum = np.sum(chroma_cqt, axis=1)
        g = ha_sum / np.max(ha_sum)

        if is_extract:
            h = meloext.extract(y, sr, is_decompose)
            return h, g

        if feature == 'chroma':
            h = chroma_cqt
        if feature == 'cqt':
            h = librosa.cqt(y, sr=sr)

        return h, g
