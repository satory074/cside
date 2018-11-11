import os
import warnings
import librosa
import numpy as np
import audf
import cqt
warnings.filterwarnings('ignore')


class Song:
    def __init__(self, path, feat):
        self.name = os.path.basename(path).split(".")[0]
        self.feat = feat

        print(f"### Load {self.name}...")
        self.h, self.g, self.tempo = self._audio2feature(self._load_song(path, feat))
        _, self.len_ = np.shape(self.h)

    def _load_song(self, path, feat):
        y, sr = librosa.load(path)
        onset = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset, sr=sr, start_bpm=180.)

        return y, sr, float(tempo[0])

    def _audio2feature(self, song_info):
        y, sr, tempo = song_info

        if self.feat == 'cqt':
            h = cqt.chroma_cqt(
                y=y, sr=sr, fmin=12, fmax=72, tempo=tempo, feature=self.feat)

            ha_sum = np.sum(h, axis=1)
            g = ha_sum / np.max(ha_sum)

            return h, g, tempo

        if self.feat == 'audf':
            audf_ = audf.AudioFingerPrint()
            fp = audf_.wavfile2peaks(y, sr)

            return fp, None, tempo
