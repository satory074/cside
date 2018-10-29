import numpy as np

class Song:
    def __init__(self, path, feature):
        import os

        self.name = os.path.basename(path).split(".")[0]
        self.h, self.g, self.tempo = self._path2g(path, feature)
        self.len_ = self.h.shape[1]

    def _path2g(self, path, feature):
        import librosa, cqt
        print (f"\n### Load {self.name}...")

        # extract chroma
        y, sr = librosa.load(path)

        onset = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset, sr=sr, start_bpm=180)

        y_harmonic, y_percussive = librosa.effects.hpss(y)
        h = cqt.chroma_cqt(y=y, sr=sr, fmin=12, fmax=48,
            tempo=tempo, feature=feature)

        ha_sum = np.sum(h, axis=1)
        g = ha_sum / np.max(ha_sum)

        return h, g, float(tempo)
