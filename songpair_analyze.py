import numpy as np

import song_analyze

class SongPair:
    def __init__(self, paths, feature, cpr, matrices):
        self.song1 = song_analyze.Song(paths[0], feature)
        self.song2 = song_analyze.Song(paths[1], feature)

        self.oti = self._calcOTI(self.song1.g, self.song2.g)
        #self.oti = 0

        self.song1.htr = np.roll(self.song1.h, self.oti, axis=0)
        print

        print "[" + self.song1.filename + "][" + self.song2.filename + "]"
        self.crp_R = self._calc_R(self.song1.htr, self.song2.h, cpr)

        if 'l' in matrices:
            self.crp_L, self.Lmax = self._calc_L()
        if 's' in matrices:
            self.crp_S, self.Smax = self._calc_SQ()
        if 'q' in matrices:
            self.crp_Q, self.Qmax = self._calc_SQ(gamma_o=5, gamma_e=0.5)


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
            calc_mat.append(np.where(n1 < epsiron, 1, 0))

        return calc_mat

    def _calc_R(self, X1, X2, cpr):
        print ("### matrix R calculate...")
        print
        crp_R = []

        smm = []
        for n1 in X1.T:
            row = []
            for n2 in X2.T:
                row.append(np.linalg.norm((n2 - n1), ord=1))
            smm.append(row)

        row_heviside = self._calchev(smm, cpr)
        col_heviside = self._calchev(np.array(smm).T, cpr)

        for i in range(len(row_heviside)):
            row = []
            for j in range(len(row_heviside[0])):
                row.append(row_heviside[i][j] * col_heviside[j][i])
            crp_R.append(row)

        #crp_R = crp_R[::-1]

        return crp_R

    def _calc_L(self):
        print ("### matrix L calculate...")
        crp_L = np.array(self.crp_R)
        crp_L[:,:] = 0

        for i in range(1, len(crp_L)):
            for j in range(1, len(crp_L[0])):
                crp_L[i][j] = crp_L[i-1][j-1] + 1 if self.crp_R[i][j] == 1 else 0

        Lmax = int(crp_L.max())
        print "Lmax: " + str(Lmax)
        print
        return crp_L, Lmax

    def _calc_SQ(self, gamma_o=float("inf"), gamma_e=float("inf")):
        matlabel = 'S' if gamma_o == float("inf") else 'Q'
        print ("### matrix " + matlabel + " calculate...")

        crp_SQ = np.array(self.crp_R)
        crp_SQ[:,:] = 0

        for i in range(2, len(crp_SQ)):
            for j in range(2, len(crp_SQ[0])):
                if self.crp_R[i][j] == 1:
                    crp_SQ[i][j] = max(crp_SQ[i-1][j-1], crp_SQ[i-2][j-1], crp_SQ[i-1][j-2]) + 1
                else:
                    crp_SQ[i][j] = max(0,
                    crp_SQ[i-1][j-1] - (gamma_o if self.crp_R[i-1][j-1] == 1 else gamma_e),
                    crp_SQ[i-2][j-1] - (gamma_o if self.crp_R[i-2][j-1] == 1 else gamma_e),
                    crp_SQ[i-1][j-2] - (gamma_o if self.crp_R[i-1][j-2] == 1 else gamma_e))

        SQmax = int(crp_SQ.max())
        print matlabel + "max: " + str(SQmax)
        print
        return crp_SQ, SQmax
