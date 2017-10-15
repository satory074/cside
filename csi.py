import librosa.display
import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import csv
import math

k = 0.1

# No use
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Not maintenance yet
def librosaRP(ga):
    R = librosa.segment.recurrence_matrix(ga, metric='cosine')
    R_aff = librosa.segment.recurrence_matrix(ga, mode='affinity')

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    librosa.display.specshow(R, x_axis='time', y_axis='time')
    plt.title('Binary recurrence (symmetric)')
    plt.subplot(1, 2, 2)
    librosa.display.specshow(R_aff, x_axis='time', y_axis='time', cmap='magma_r')
    plt.title('Affinity recurrence')
    plt.tight_layout()
    plt.show()

class Song:
    def __init__(self, is_recalc, path):
        # savepath ="/Users/satory43/Desktop/Programs/Python/py2/csi/data/song/"
        savepath = "/Volumes/satoriHD/data/song/"

        self.path = path
        self.filename = (path.split("/")[-1]).split(".")[0]
        self.ifp = True if is_recalc else self._is_first_processing(savepath)
        self.h, self.g = self._path2g(savepath)
        self.htr = self.h

    def _is_first_processing(self, savepath):
        for x in os.listdir(savepath + "chroma/"):
            if os.path.isfile(savepath + "chroma/" + x):
                if x.split(".")[0] == self.filename:
                    return False

        return True

    def _path2g(self, savepath):
        print ("##### " + self.filename + " load...")

        if self.ifp:
            # extract chroma
            y, sr = librosa.load(self.path)
            chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

            ha_sum = np.sum(chroma_cq, axis=1)
            g = ha_sum / np.max(ha_sum)

            # output
            np.savetxt(savepath + "chroma/" + self.filename + ".csv", chroma_cq, delimiter=',')
            np.savetxt(savepath + "exchroma/" + self.filename + ".csv", g, delimiter=',')

            return chroma_cq, g
        else:
            # finding same file
            chroma_cq = np.loadtxt(savepath + "chroma/" + self.filename + ".csv", delimiter=',')
            g = np.loadtxt(savepath + "exchroma/" + self.filename + ".csv", delimiter=',')

            return chroma_cq, g

class SongPair:

    def __init__(self, is_recalc, s1, s2):
        #savepath ="/Users/satory43/Desktop/Programs/Python/py2/csi/data/songpair/"
        savepath = "/Volumes/satoriHD/data/songpair/"        

        songs = [s1.filename, s2.filename]
        songs.sort()
        Song1 = s1 if s1.filename == songs[0] else s2
        Song2 = s2 if s2.filename == songs[1] else s1

        self.filename = Song1.filename + "_" + Song2.filename
        self.ifp = True if is_recalc else self._is_first_processing(savepath)
        self.oti = self._calcOTI(Song1.g, Song2.g, savepath)

        Song1.htr = np.roll(Song1.h, -self.oti, axis=0)

        self.crp_R = self._calc_R(Song1.htr, Song2.h, savepath)
        self.crp_L, self.Lmax = self._calc_L(savepath)

    def _is_first_processing(self, savepath):
        for x in os.listdir(savepath + "oti/"):
            if os.path.isfile(savepath + "oti/" + x):
                if x.split(".")[0] == self.filename:
                    return False

        return True

    def _calcOTI(self, ga, gb, savepath):
        oti = 0

        if self.ifp:
            csgb = gb
            dots = []
            for i in range(12):
                dots.append(np.dot(ga, csgb))
                csgb = np.roll(csgb, -1)

            oti =  np.argmax(dots)

            f = open(savepath + "oti/" + self.filename + '.txt', 'w')
            f.write(str(oti))
            f.close()
        else:
            f = open(savepath + "oti/" + self.filename + '.txt')
            oti = int(f.read())
            f.close()

        print "OTI: " + str(oti)
        return oti

    def _calchev(self, mat):
        calc_mat = []
        for n1 in mat:
            sortedlist = np.sort(n1)
            epsiron = sortedlist[int(len(sortedlist) * k)]
            calc_mat.append(np.where(n1 < epsiron, 1, 0))

        return calc_mat

    def _calc_R(self, X1, X2, savepath):
        print ("##### " + self.filename + " matrix R calculate...")
        crp_R = []

        if self.ifp:
            smm = []
            for n1 in X1.T:
                row = []
                for n2 in X2.T:
                    row.append(np.linalg.norm((n2 - n1), ord=1))
                smm.append(row)

            row_heviside = self._calchev(smm)
            col_heviside = self._calchev(np.array(smm).T)

            for i in range(len(row_heviside)):
                row = []
                for j in range(len(row_heviside[0])):
                    row.append(row_heviside[i][j] * col_heviside[j][i])
                crp_R.append(row)

            crp_R = crp_R[::-1]

            np.savetxt(savepath + "crp_R/" + self.filename + ".csv", crp_R, delimiter=',')
        else:
            crp_R = np.loadtxt(savepath + "crp_R/" + self.filename + ".csv", delimiter=',')

        return crp_R

    def _calc_L(self, savepath):
        print ("##### " + self.filename + " matrix L calculate...")
        crp_L = np.array(self.crp_R)

        if self.ifp:
            crp_L[:,:] = 0

            for i in range(1, len(crp_L)):
                for j in range(1, len(crp_L[0])):
                    crp_L[i][j] = crp_L[i-1][j-1] + 1 if self.crp_R[i][j] == 1 else 0

            np.savetxt(savepath + "crp_L/" + self.filename + ".csv", crp_L, delimiter=',')
        else:
            crp_L = np.loadtxt(savepath + "crp_L/" + self.filename + ".csv", delimiter=',')

        Lmax = int(crp_L.max())
        f = open(savepath + "Lmax/" + self.filename + '.txt', 'w')
        f.write(str(Lmax))
        f.close()
        print "Lmax: " + str(Lmax)
        return crp_L, Lmax

    def draw_heatmap(self, list_, xlabel="", ylabel=""):
        data = np.array(list_)
        sns.heatmap(data, xticklabels=True, yticklabels=True, cmap="Blues")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        print ("##### heatmap showing...")
        plt.show()


def main(is_recalc, path1, path2):
    song1 = Song(is_recalc, path1)
    song2 = Song(is_recalc, path2)

    songpair = SongPair(is_recalc, song1, song2)
    # songpair.draw_heatmap(songpair.crp_R, song1.filename, song2.filename)
    # songpair.draw_heatmap(songpair.crp_L, song1.filename, song2.filename)

def main2():
    PATH = "/Users/satory43/Desktop/Programs/data/coversongs/covers32k/"
    path1 = "/Users/satory43/Desktop/Programs//data/coversongs/covers32k/Yesterday/beatles+1+11-Yesterday.mp3"
    song1 = Song(True, path1)

    for pdir in os.listdir(PATH):
        pdirname = PATH + pdir
        if os.path.isdir(pdirname):
            for cdir in os.listdir(pdirname):
                cdirname = pdirname + "/" + cdir
                if os.path.isfile(cdirname):
                    path2 = cdirname
                    song2 = Song(True, path2)

                    songpair = SongPair(True, song1, song2)
                print ""
        print "---"




if __name__ == '__main__':
    #is_recalc = True if sys.argv[1]=='-recalc' else False
    #main(is_recalc, sys.argv[2], sys.argv[3])
    main2()
