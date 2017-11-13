import librosa.display
import sys
import os

import numpy as np
from scipy import stats
import csv
import math
import re

import song_analyze
import draw_heatmap


k = 0.1

# No use
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


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
        print ("is_first_processing: " + str(self.ifp))
        self.oti = self._calcOTI(Song1.g, Song2.g, savepath)

        Song1.htr = np.roll(Song1.h, self.oti, axis=0)

        self.crp_R = self._calc_R(Song1.htr, Song2.h, savepath)
        self.crp_L, self.Lmax = self._calc_L(savepath)
        self.crp_S, self.Smax = self._calc_S(savepath)
        self.crp_Q, self.Qmax = self._calc_Q(savepath)

        f = open(savepath + 'LSQmax/' + self.filename + '.txt', 'w')
        f.write(str(self.Lmax) + ", " + str(self.Smax) + ", " + str(self.Qmax))
        f.close()

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

            #crp_R = crp_R[::-1]

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
        print "Lmax: " + str(Lmax)
        return crp_L, Lmax

    def _calc_S(self, savepath):
        print ("##### " + self.filename + " matrix S calculate...")
        crp_S = np.array(self.crp_R)

        if self.ifp:
            crp_S[:,:] = 0

            for i in range(2, len(crp_S)):
                for j in range(2, len(crp_S[0])):
                    if self.crp_R[i][j] == 1:
                        crp_S[i][j] = max(crp_S[i-1][j-1], crp_S[i-2][j-1], crp_S[i-1][j-2]) + 1
                    else:
                        crp_S[i][j] = 0

            np.savetxt(savepath + "crp_S/" + self.filename + ".csv", crp_S, delimiter=',')
        else:
            crp_S = np.loadtxt(savepath + "crp_S/" + self.filename + ".csv", delimiter=',')

        Smax = int(crp_S.max())
        print "Smax: " + str(Smax)
        return crp_S, Smax

    def _calc_Q(self, savepath):
        print ("##### " + self.filename + " matrix Q calculate...")
        crp_Q = np.array(self.crp_R)
        gamma_o = 5
        gamma_e = 0.5

        if self.ifp:
            crp_Q[:,:] = 0

            for i in range(2, len(crp_Q)):
                for j in range(2, len(crp_Q[0])):
                    if self.crp_R[i][j] == 1:
                        crp_Q[i][j] = max(crp_Q[i-1][j-1], crp_Q[i-2][j-1], crp_Q[i-1][j-2]) + 1
                    else:
                        crp_Q[i][j] = max(0,
                        crp_Q[i-1][j-1] - (gamma_o if self.crp_R[i-1][j-1] == 1 else gamma_e),
                        crp_Q[i-2][j-1] - (gamma_o if self.crp_R[i-2][j-1] == 1 else gamma_e),
                        crp_Q[i-1][j-2] - (gamma_o if self.crp_R[i-1][j-2] == 1 else gamma_e))

            np.savetxt(savepath + "crp_Q/" + self.filename + ".csv", crp_Q, delimiter=',')
        else:
            crp_Q = np.loadtxt(savepath + "crp_Q/" + self.filename + ".csv", delimiter=',')

        Qmax = int(crp_Q.max())
        print "Qmax: " + str(Qmax)
        return crp_Q, Qmax


def main(is_recalc, path1, path2):
    song1 = song_analyze.Song(is_recalc, path1)
    song2 = song_analyze.Song(is_recalc, path2)

    songpair = SongPair(is_recalc, song1, song2)
    draw_heatmap.draw(songpair.crp_R, xlabel=song2.filename, ylabel=song1.filename,
                        x_axis='time', y_axis='time')
    #songpair.draw_heatmap(songpair.crp_L, song1.filename, song2.filename)
    #songpair.draw_heatmap(songpair.crp_S, song1.filename, song2.filename)
    #songpair.draw_heatmap(songpair.crp_Q, song1.filename, song2.filename)

def main2(is_recalc, PATH, path1):
    #PATH = "/Users/satory43/Desktop/Programs/data/csi_test/"
    #path1 = "/Users/satory43/Desktop/Programs//data/csi_test/01_120.mp3"
    song1 = song_analyze.Song(is_recalc, path1)

    for pdir in os.listdir(PATH):
        pdirname = PATH + pdir
        if os.path.isfile(pdirname):
            #for cdir in os.listdir(pdirname):
            #    cdirname = pdirname + "/" + cdir
            #    if os.path.isfile(cdirname):
            #        path2 = cdirname
            #        song2 = Song(True, path2)

            #        songpair = SongPair(True, song1, song2)
            #    print ""

            if os.path.splitext(pdirname)[1] == ".mp3":
                path2 = pdirname
                song2 = Song(is_recalc, path2)

                songpair = SongPair(is_recalc, song1, song2)
        print "---"


if __name__ == '__main__':
    is_recalc = True if sys.argv[1]=='-recalc' else False
    main(is_recalc, sys.argv[2], sys.argv[3])
    #main2(is_recalc, sys.argv[2], sys.argv[3])
