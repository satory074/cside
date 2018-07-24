import glob
import numpy as np
import os
import pandas as pd
from scipy import signal
import sys

def main(path="output/Qmaxlist/"):
    files = glob.glob(path + '/*.txt')

    Qmaxlist = []
    for file_ in files:
        f = open(file_, "r")
        list_ = []
        for x in f:
            list_.append(float(x.rstrip("\n")))
        f.close()


        reducelist = []
        pos0 = 0
        nsimseg = 1
        for i in range(2247):
            nsimseg = 1 if list_[i] == 0.0 else nsimseg + 1
            reducelist.append(list_[i] if i < np.argmax(list_) else 0.0)

        Qmaxlist.append(reducelist)

    df = pd.DataFrame(Qmaxlist)
    df.to_csv("output/Qmaxlist/qmaxlist.csv")

    medQmax = []
    np_ = Qmaxlist
    for i in range(len(Qmaxlist[0])):
        list_ = []
        for j in range(len(Qmaxlist)):
            list_.append(Qmaxlist[j][i])

        #medQmax.append(np.argmax(np.array(Qmaxlist)[:][i]))
        medQmax.append(np.argmax(np.array(list_)))


    seg_continue = 0
    for i in range(1, len(medQmax)):
        seg_continue += 1
        if medQmax[i] != medQmax[i-1]:
            print (i, medQmax[i])



if __name__ == '__main__':
    main(sys.argv[1])
