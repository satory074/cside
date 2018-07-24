# coding: utf-8
import numpy as np

import song_analyze

class SongPair:
    def __init__(self, paths, feature, oti=0):
        print ("### Load songs")
        self.song1 = song_analyze.Song(paths[0], feature)
        self.song2 = song_analyze.Song(paths[1], feature)
        self.filename = self.song1.filename + "_" + self.song2.filename

        #self.oti = self._calcOTI(self.song1.g, self.song2.g)
        self.oti = oti
        print("OTI: {}".format(self.oti))
        self.song1.h = np.roll(self.song1.h, self.oti, axis=0)

        self.crp_R = self._calc_R(self.song1.h, self.song2.h)
        self.crp_Q, self.Qmax, self.Qmaxlist, self.segends_Q = self._calc_Q()

    def _calcOTI(self, ga, gb):
        # TODO: CQT processing
        csgb = gb
        dots = []
        for i in range(12):
            dots.append(np.dot(ga, csgb))
            csgb = np.roll(csgb, -1)

        return np.argmax(dots)

    def _calchev(self, mat, cpr):
        calc_mat = []
        for v in mat:
            sortedlist = np.sort(v)
            epsiron = sortedlist[int(sortedlist.shape[0] * cpr)]
            calc_mat.append(np.where(v <= epsiron, 1, 0))

        return np.array(calc_mat)

    def _calc_R(self, X1, X2, cpr=0.1):
        print ("### Calculate matrix R...")

        crp_R = []
        smm = []
        for n1 in X1.T:
            #n1 = np.where(n1 == np.max(n1), np.max(n1), 0.0)
            smm.append([np.linalg.norm((n2 - n1), ord=1) for n2 in X2.T])

        rhev = self._calchev(smm, cpr)
        chev = self._calchev(np.array(smm).T, cpr)

        row, col = rhev.shape
        for i in range(row):
            crp_R.append([rhev[i][j] * chev[j][i] for j in range(col)])
            if i % 500 == 0: print ("\t{}/{}".format(i, row))

        return crp_R
        #return crp_R[::-1]

    def _calc_Q(self, ga_o=5.0, ga_e=0.5):
        print ("### Calculate matrix Q...")

        row, col = np.shape(self.crp_R)
        crp_Q = np.zeros((row, col))
        for i in range(2, row):
            for j in range(2, col):
                if self.crp_R[i][j] == 1:
                    look = [crp_Q[i-1][j-1], crp_Q[i-2][j-1], crp_Q[i-1][j-2]]
                    crp_Q[i][j] = max(look) + 1.0
                else:
                    eq = lambda x, y: \
                        crp_Q[x][y] - (ga_o if self.crp_R[x][y] == 1 else ga_e)
                    look = [0, eq(i-1, j-1), eq(i-2, j-1), eq(i-1, j-2)]
                    crp_Q[i][j] = max(look)

            if i % 500 == 0: print("\t{}/{}".format(i, row))

        Qmax = crp_Q.max()
        Qmaxlist = [max(row) for row in crp_Q]
        segends = [tuple(list_) for list_ in np.argwhere(crp_Q == Qmax)]

        #f = open("output/Qmaxlist/{}.txt".format(self.filename), 'w')
        #f.write("{}\n".format([x for x in Qmaxlist]))
        #f.close()

        print ("\tQmax: {}".format(Qmax))
        print ("\tQmax end: {}".format(segends))

        return crp_Q, Qmax, Qmaxlist, segends

    def illustrate_matrix(self):
        import matplotlib as plt
