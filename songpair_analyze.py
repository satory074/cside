# coding: utf-8
import itertools as it
import numpy as np

import draw_heatmap
import song_analyze as songanal

class SongPair:
    def __init__(self, medley, songpath, feature, lwin, oti):
        self.medley = medley
        self.song = songanal.Song(path=songpath, feature=feature)
        self.filename = "{}_{}".format(self.medley.name, self.song.name)

        #self.oti = self._calcOTI(self.song1.g, self.song2.g)
        self.medley.h = np.roll(self.medley.h, oti, axis=0)

        self.R = self._calc_R(self.medley, self.song, lwin)
        self.Q, self.Qmaxlist, self.segends_Q = self._calc_Q(self.R)

    def _calcOTI(self, ga, gb):
        csgb = gb
        dots = []
        for i in range(12):
            dots.append(np.dot(ga, csgb))
            csgb = np.roll(csgb, -1)

        #print("\tOTI: {}".format(np.argmax(dots)))

        return np.argmax(dots)

    def _calchev(self, mat, cpr):
        calc_mat = []
        for v in mat:
            sortedlist = np.sort(v)
            epsiron = sortedlist[int(sortedlist.shape[0] * cpr)]
            calc_mat.append(np.where(v <= epsiron, 1, 0))

        return np.array(calc_mat)

    def _cnglwin(self, smm, tempo, lwin, sr=22050, hop_length=512.):
        print ("\t\t### Change window length...")
        spb = lambda x: ((60 / x) * (sr / hop_length)) * lwin #samples per beat
        tempox, tempoy = tempo
        rhop = int(spb(tempox))
        chop = int(spb(tempoy))

        row, col = smm.shape
        smm_note = []
        for i in range(0, row-rhop, rhop):
            smm_note.append([smm[i:i+rhop+1, j:j+chop+1].min() \
                for j in range(0, col-chop, chop)])

        return np.array(smm_note)

    def _calc_R(self, medley, song, lwin, cpr=0.1):
        print ("\t### Calculate matrix R...")

        matshape = (medley.len_, song.len_)
        tempo = (medley.tempo, song.tempo)

        from scipy.spatial import distance as dist
        smm = np.array([dist.euclidean(vecm, vecs) \
                for vecm, vecs in it.product(medley.h.T, song.h.T)])
        smm = np.reshape(smm, matshape)
        smm_note = self._cnglwin(smm, tempo, lwin)

        return self._calchev(smm_note, cpr) * self._calchev(smm_note.T, cpr).T
        #return crp_R[::-1]

    def _calc_Q(self, R, ga_o=5.0, ga_e=0.5):
        print ("\t### Calculate matrix Q...")

        row, col = np.shape(R)
        Q = np.zeros((row, col))
        for i, j in it.product(range(2, row), range(2, col)):
            if R[i][j] == 1:
                Q[i][j] = max([Q[i-1][j-1], Q[i-2][j-1], Q[i-1][j-2]]) + 1.0
            else:
                eq = lambda x, y: Q[x][y] - (ga_o if R[x][y] == 1 else ga_e)
                Q[i][j] = max([0, eq(i-1, j-1), eq(i-2, j-1), eq(i-1, j-2)])

        Qmaxlist = [max(row) for row in Q]
        segends = [tuple(list_) for list_ in np.argwhere(Q == Q.max())]

        print ("\tQmax: {}".format(Q.max()))
        #print ("\tQmax end: {}".format(segends))

        return Q, Qmaxlist, segends
