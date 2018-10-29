#coding: utf-8
"""
Music Segments Detector for Medley

Usage:
    muse_de_form.py <dir_med> [options]
    muse_de_form.py (<file_med> <file_song>) [options]

Options:
    -f <feature>, --feature <feature>    Used for features that generate CRP [default: cqt]
    -l <len>, --length <len>             number of half notes [default: 0.5]
    -o <oti>, --oti <oti>                OTI [default: 0]
    --help                               Show this help message and exit
"""

from docopt import docopt
import os
import sys

import numpy as np
import pandas as pd

import song_analyze as songanal
import songpair_analyze as pairanal

def load_oti(path):
    otilist = []
    with open(path) as f:
        for d in f.readlines():
            otilist.append(tuple(d.split(', ')))

    return otilist

def save_data(songpairs):
    data = pd.DataFrame([],
    columns=['Name', 'length', 'Name', 'length','Qmax',
    #'segends_Q'
    ])
    d = []
    for i in np.arange(len(songpairs[0].Qmaxlist)):
        d.append(str(i))
    qmax = []
    index = []
    id = 0
    for songpair in songpairs:
        id += 1
        data.loc[id] = [songpair.medley.name,
                        songpair.Q.shape[0],
                        songpair.song.name,
                        songpair.Q.shape[1],
                        #songpair.oti,
                        #songpair.feature,
                        songpair.Q.max(),
                        #songpair.segends_Q
                        ]
        qmax.append(songpair.Qmaxlist)
        index.append(songpair.song.name)

    data.to_csv(f"output/intensive/{songpair.medley.name}.csv")
    pd.DataFrame(qmax, index=index).to_csv(f"output/Qmaxlist/{songpair.medley.name}")

def main(argv):
    ### args ###
    args = docopt(__doc__)
    dir_med = args['<dir_med>']
    file_med, file_song = [args['<file_med>'], args['<file_song>']]
    feature = args['--feature']
    lwin = args['--length']
    oti = args['--oti']

    print(f"[feature] {feature}")
    print(f"[lwin] {lwin}")

    ###  ###
    if dir_med:
        file_med = (f"{dir_med}/{os.path.basename(dir_med)}.mp3")
        medley = songanal.Song(path=file_med, feature=feature)
        li_songpair = [pairanal.SongPair(medley=medley,
                                        songpath=(f"{dir_med}/{name}"),
                                        feature=feature,
                                        lwin=float(lwin),
                                        oti=int(oti))
            for name, oti in load_oti(f"{dir_med}/oti.txt")]
    else:
        li_songpair = [pairanal.SongPair(medley=file_med,
                                        songpath=file_song,
                                        feature=feature,
                                        lwin=float(lwin),
                                        oti=int(oti))]

    save_data(li_songpair)

if __name__ == '__main__':
    main(sys.argv)
