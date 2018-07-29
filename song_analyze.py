import numpy as np
#import melody_extraction as meloext

class Song:
    def __init__(self, path, feature):
        import os

        self.name = os.path.basename(path).split(".")[0]
        self.h, self.g, self.tempo = self._path2g(path, feature)
        self.len_ = self.h.shape[1]

    def _path2g(self, path, feature):
        import librosa, cqt
        print ("### Load {}...".format(self.name))

        # extract chroma
        y, sr = librosa.load(path)

        onset = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset, sr=sr, start_bpm=180)
        print("\ttempo: {}".format(tempo))

        y_harmonic, y_percussive = librosa.effects.hpss(y)
        h = cqt.chroma_cqt(y=y_harmonic, sr=sr,
            tempo=tempo, feature=feature,fmin=24, fmax=72
        )

        ha_sum = np.sum(h, axis=1)
        g = ha_sum / np.max(ha_sum)

        return h, g, float(tempo)
