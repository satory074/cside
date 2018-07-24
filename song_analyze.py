# coding: utf-8
import librosa
import numpy as np
import os

import cqt
#import melody_extraction as meloext

class Song:
    def __init__(self, path, feature):
        self.path = path
        self.feature = feature
        self.filename = os.path.basename(path).split(".")[0]
        self.h, self.g = self._path2g(feature)

    def _path2g(self, feature):
        print ("\t### Load {}...".format(self.filename))

        # extract chroma
        y, sr = librosa.load(self.path)

        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, start_bpm=180)
        print("\t\ttempo: {}".format(float(tempo)))

        y_harmonic, y_percussive = librosa.effects.hpss(y)
        h = cqt.chroma_cqt(y=y_harmonic, sr=sr, tempo=tempo, lwintype='note', threshold=0.0, feature=feature)

        ha_sum = np.sum(h, axis=1)
        g = ha_sum / np.max(ha_sum)

        return h, g
