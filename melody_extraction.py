# coding: utf-8
import librosa
import numpy as np
import matplotlib.pyplot as plt
import sys
import vamp

def hz2midi(hz):
    # convert from Hz to midi note
    hz_nonneg = hz.copy()
    hz_nonneg[hz <= 0] = 0
    midi = 69 + 12*np.log2(hz_nonneg/440.)
    midi[midi <= 0] = 0

    # round
    midi = np.round(midi)

    return midi

def extract(y, sr, is_decompose):
    #y, sr = librosa.load(path, mono=True)
    if is_decompose:
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        y = y_harmonic
    melody = vamp.collect(y, sr, "mtg-melodia:melodia",
                        parameters={"voicing": 0.2})

    pitch = melody['vector'][1][::8]
    #pitch = np.insert(pitch, 0, [0]*8)

    midi_pitch = hz2midi(pitch)
    midi_pitch = midi_pitch % 12

    return midi_pitch.reshape(1, len(midi_pitch))

if __name__ == '__main__':
    main(sys.argv[1])
