# coding: utf-8
# source: https://github.com/justinsalamon/audio_to_midi_melodia.git
import librosa
import numpy as np
import matplotlib.pyplot as plt
import sys
import vamp

def maxElem(list_):
    # source: http://peroon.hatenablog.com/entry/20091222/1261505808
    '''
    与えられたリストの中に、最も多く存在する要素を返す
    (最大の数の要素が複数ある場合、pythonのsetで先頭により近い要素を返す)
    '''
    L = list_[:]#copy
    S = set(list_)
    S = list(S)
    MaxCount = 0
    ret = 'nothing...'

    for elem in S:
        c = 0
        while elem in L:
            ind = L.index(elem)
            foo = L.pop(ind)
            c += 1

        if c > MaxCount:
            MaxCount = c
            ret = elem

        return ret

def hz2midi(hz):
    # convert from Hz to midi note
    hz_nonneg = hz.copy()
    hz_nonneg[hz <= 0] = 0
    midi = 69 + 12*np.log2(hz_nonneg/440.)
    midi[midi <= 0] = 0

    # round
    midi = np.round(midi)

    return midi

def extract(y, sr):
    melody = vamp.collect(y, sr, "mtg-melodia:melodia",
                        parameters={"voicing": 2.0})

    pitch = []
    locpitchs = []
    for i, p in enumerate(melody['vector'][1]):
        locpitchs.append(p)

        if i % 8 == 0:
            pitch.append(maxElem(locpitchs))
            locpitchs = []

    midi_pitch = hz2midi(np.array(pitch))
    midi_pitch = midi_pitch % 12

    return midi_pitch.reshape(1, len(midi_pitch))

if __name__ == '__main__':
    main(sys.argv[1])
