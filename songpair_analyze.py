# coding: utf-8
import numpy as np

import song_analyze

class SongPair:
    def __init__(self, paths, feature, oti):
        self.CPR = 0.1 # CRP plot rate

        self.song1 = song_analyze.Song(paths[0], feature)
        self.song2 = song_analyze.Song(paths[1], feature)
        self.filename = self.song1.filename + "_" + self.song2.filename

        #self.oti = self._calcOTI(self.song1.g, self.song2.g)
        self.oti = oti
        print(self.oti)

        self.song1.htr = np.roll(self.song1.h, self.oti, axis=0)
        print ("[{}]-[{}]".format(self.song1.filename, self.song2.filename))

        self.crp_R = self._calc_R(self.song1.htr, self.song2.h, self.CPR)
        self.crp_Q, self.Qmax, self.segends_Q = self._calc_Q()

    def _calcOTI(self, ga, gb):
        oti = 0

        csgb = gb
        dots = []
        for i in range(12):
            dots.append(np.dot(ga, csgb))
            csgb = np.roll(csgb, -1)

        oti =  np.argmax(dots)

        print "OTI: " + str(oti)
        return oti

    def _calchev(self, mat, cpr):
        calc_mat = []
        for n1 in mat:
            sortedlist = np.sort(n1)
            epsiron = sortedlist[int(len(sortedlist) * cpr)]
            calc_mat.append(np.where(n1 <= epsiron, 1, 0))

        return calc_mat

    def _calc_R(self, X1, X2, cpr):
        print ("### matrix R calculate...")
        print
        crp_R = []

        smm = []
        for n1 in X1.T:
            row = []
            n1 = np.where(n1 == np.max(n1), np.max(n1), 0.0)
            for n2 in X2.T:
                n2 = np.where(n2 == np.max(n2), np.max(n2), 0.0)
                row.append(np.linalg.norm((n2 - n1), ord=1))
            smm.append(row)

        row_heviside = self._calchev(smm, cpr)
        col_heviside = self._calchev(np.array(smm).T, cpr)

        for i in range(len(row_heviside)):
            row = []
            for j in range(len(row_heviside[0])):
                row.append(row_heviside[i][j] * col_heviside[j][i])
            crp_R.append(row)

            if i % 500 == 0:
                print (str(i) + "/" + str(len(row_heviside)))

        #crp_R = crp_R[::-1]

        return crp_R

    def _calc_Q(self, gamma_o=5.0, gamma_e=0.5):
        matlabel = 'S' if gamma_o == float("inf") else 'Q'
        print ("### Calculate matrix S...")

        rrow, rcol = np.shape(self.crp_R)

        crp_Q = np.zeros((rrow, rcol))

        for i in range(2, rrow):
            for j in range(2, rcol):
                if self.crp_R[i][j] == 1:
                    arglook = [(i-1, j-1), (i-2, j-1), (i-1, j-2)]
                    look = [crp_Q[i-1][j-1], crp_Q[i-2][j-1], crp_Q[i-1][j-2]]

                    max_ = max(look)
                    crp_Q[i][j] = max_ + 1.0
                else:
                    arglook = [(None, None), (i-1, j-1), (i-2, j-1), (i-1, j-2)]
                    look = [0,
                    crp_Q[i-1][j-1] - (gamma_o if self.crp_R[i-1][j-1] == 1 else gamma_e),
                    crp_Q[i-2][j-1] - (gamma_o if self.crp_R[i-2][j-1] == 1 else gamma_e),
                    crp_Q[i-1][j-2] - (gamma_o if self.crp_R[i-1][j-2] == 1 else gamma_e)]

                    max_ = max(look)
                    crp_Q[i][j] = max_

            if i % 500 == 0:
                print (str(i) + "/" + str(rrow))

        segends = []
        Qmax = crp_Q.max()

        Qmaxlist = []
        for row in crp_Q:
            Qmaxlist.append(max(row))
        print(len(Qmaxlist))
        f = open("output/Qmaxlist/" + self.filename + '.txt', 'w')
        for x in Qmaxlist:
            f.write(str(x) + "\n")
        f.close()


        for list_ in np.argwhere(crp_Q == Qmax):
            x, y = list_
            segends.append((x, y))

        print matlabel + "max: " + str(Qmax)
        print matlabel + "end: " + str(segends)
        print
        return crp_Q, Qmax, segends

    def illustrate_matrix(self):
        import matplotlib as plt
