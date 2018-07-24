import glob
import os
import pandas as pd
import sys


def main(path="output/segdata/"):
    files = glob.glob(path + '/*.txt')

    data = []
    for file_ in files:
        f = open(file_, "r")
        list_ = []
        for x in f:
            list_.append(x.rstrip("\n"))
        f.close()
        data.append(list_)
    df = pd.DataFrame(data,
    columns=['Name', 'length', 'Name', 'length',
            'OTI', 'Feature', 'is_decompose', 'is_extract',
            'Lmax', 'segstarts_L', 'segends_L',
            'Smax', 'segstarts_S', 'segends_S',
            'Qmax', 'segstarts_Q', 'segends_Q'])
    df.to_csv("output/intensive/kumikyokuAdur1.csv")

if __name__ == '__main__':
    main(sys.argv[1])
