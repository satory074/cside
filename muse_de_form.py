#coding: utf-8
"""
Music Segments Detector for Medley

usage: muse_de_form.py (<medley> <songdir>) [options]

options:
    -d, --decompose                      Decompose an audio time series into harmonic and percussive components.
    -e, --extract                        Do melody extract
    -f <feature>, --feature <feature>    Used for features that generate CRP [default: cqt]
    -o <oti>, --oti <oti> [default: 0]
    --help                               Show this help message and exit
"""

import csv
from docopt import docopt
import numpy as np
import os
import pandas as pd
import sys

import draw_heatmap
import song_analyze
import songpair_analyze

def load_oti(path):
    otilist = []
    with open(path) as f:
        for d in f.readlines():
            otilist.append(tuple(d.split(', ')))

    return otilist



def save_data(songpairs):
    #draw_heatmap.draw(songpair.crp_R, xlabel=songpair.song2.filename, ylabel=songpair.song1.filename,
    #                    x_axis='time', y_axis='time')

    data = pd.DataFrame([],
    columns=['Name', 'length', 'Name', 'length',
    'oti', 'feature','Qmax',
    #'segends_Q'
    ])
    #d = ["name"]
    d = []
    for i in np.arange(len(songpairs[0].Qmaxlist)):
        d.append(str(i))
    qmax = []
    index = []
    id = 0
    for songpair in songpairs:
        id += 1
        data.loc[id] = [songpair.song1.filename,
                        songpair.crp_R.shape[0],
                        songpair.song2.filename,
                        songpair.crp_R.shape[1],
                        songpair.oti,
                        songpair.feature,
                        songpair.Qmax,
                        #songpair.segends_Q
                        ]
        qmax.append(songpair.Qmaxlist)
        index.append(songpair.song2.filename)

    data.to_csv("output/intensive/{}.csv".format(songpair.song1.filename))
    pd.DataFrame(qmax, index=index).to_csv("output/Qmaxlist/{}.csv".format(songpair.song1.filename))
    #with open("output/Qmaxlist/{}.csv".format(songpair.song1.filename), 'w') as f:
    #    writer = csv.writer(f, lineterminator='\n')
    #    writer.writerows(qmax)


def main(argv):
    args = docopt(__doc__)
    medpath, dir = [args['<medley>'], args['<songdir>']]
    feature = args['--feature']
    if args['--oti'] is not None: oti = int(args['--oti'])
    print ("[feature] {}\n".format(feature))

    songpairs = []
    medley = song_analyze.Song(medpath, feature)
    for name, oti in load_oti("{}oti.txt".format(dir)):
        print
        songpairs.append(
            songpair_analyze.SongPair(medley, (dir + name), feature, oti)
        )
    save_data(songpairs)

if __name__ == '__main__':
    main(sys.argv)
