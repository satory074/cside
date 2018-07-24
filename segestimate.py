import librosa
import numpy as np
import os
import re
import sys

class ComponentSong:
    def __init__(self, id, name, dir):
        self.id = id
        self.name = name
        self.dir = dir

        self.qmaxlist = self._load_qmaxlist(name)

    def _load_qmaxlist(self, name):

        qmaxlist = []
        f = open(self.dir)
        lines = f.readlines()
        f.close()

        qmaxlist = [float(line) for line in lines]

        return np.array(qmaxlist)

    def calc_qtotal(self, start, end):
        list_ = self.qmaxlist[start:end]
        sum = np.sum((np.roll(list_, -1) - list_)[:-1])

        return sum


def path2complist(path):
    complist = []
    id = 0

    print
    print ("### LOAD ###")
    for file in os.listdir(path):
        name, ext = os.path.splitext(file)

        if ext == '.txt':
            print(file)

            complist.append(ComponentSong(id, name, (path + '/' + file)))
            id += 1
    print

    return complist

def load_medley(path):
    y, sr = librosa.load(path)
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, start_bpm=180)

    return y, sr, float(tempo)


def main(dir, path):
    complist = path2complist(dir)
    nframe = len(complist[0].qmaxlist)

    y, sr, tempo = load_medley(path)

    HOP_LENGTH = 512.0
    SEG_LENGTH = 4.0 # bar

    spb = int((60 / tempo) * (sr / HOP_LENGTH)) # nsamples per beat

    nslide = int(spb * SEG_LENGTH)
    start = 0
    #end = nslide
    end = 8 * 2
    results_id = []
    results_val = []

    while end < nframe:
        qtotals = np.array([comp.calc_qtotal(start, end) for comp in complist])

        results_val.append(np.max(qtotals))
        results_id.append(np.argmax(qtotals))

        start = end
        #end += nslide
        end += 8 * 2

    for id in results_id:
        print (complist[id].name)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
