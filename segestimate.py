"""
hoge

usage: segestimate.py (<outputdir>) [options]

options:
    -n <num>   Used for features that generate CRP [default: 32]
    --help     Show this help message and exit
"""


from docopt import docopt
import sys
import numpy as np
import pandas as pd


import calc_accuracy

class ComponentSong:
    def __init__(self, row):
        self.name = row[0]
        self.qmaxlist = np.array([i for i in row[1:]])

    def calc_qtotal(self, start, end):
        list_ = self.qmaxlist[start:end]
        return np.sum((np.roll(list_, -1) - list_)[:-1])


def path2complist(path):
    complist = []
    df = pd.read_csv(path)
    for i in range(df.shape[0]):
        complist.append(ComponentSong(np.array(df.loc[i])))

    return complist

def estimate(nslide, complist):
    start = 0
    end = nslide
    results_id = []
    results_val = []
    nframe = len(complist[0].qmaxlist)

    while end < nframe:
        qtotals = np.array([comp.calc_qtotal(start, end) for comp in complist])

        results_val.append(np.max(qtotals))
        results_id.append(np.argmax(qtotals))

        start = end
        end += nslide

    estlist = [complist[id].name for id in results_id]
    [print(name) for name in estlist]

    return estlist


def main(argv):
    args = docopt(__doc__)
    dir = args['<outputdir>']
    nslide = int(args['-n']) # n helf note

    calc_accuracy.calc(dir, nslide, estimate(nslide, path2complist(dir)))

if __name__ == '__main__':
    main(sys.argv)
