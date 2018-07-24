# coding: utf-8
import librosa
import numpy as np
import os
import pandas as pd
import scipy.signal

import cqt
#import melody_extraction as meloext

class Song:
    def __init__(self, path, feature):
        self.path = path
        self.feature = feature
        self.filename = (os.path.splitext(path)[0]).split("/")[-1]
        self.h, self.g = self._path2g(feature)
        self.htr = self.h

    def _path2g(self, feature):
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

        if feature == 'chroma':
            h = chroma_cqt
        if feature == 'cqt':
            h = librosa.cqt(y, sr=sr)

        return h, g
