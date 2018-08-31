#coding: utf-8
"""
Music Segments Detector for Medley

usage: muse_de_form.py (<medley> <songdir>) [options]

options:
    -f <feature>, --feature <feature>    Used for features that generate CRP [default: cqt]
    -l <len>, --length <len>             number of half notes [default: 0.5]
    -o <oti>, --oti <oti>                OTI [default: 0]
    --help                               Show this help message and exit
"""

from docopt import docopt
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
    #draw_heatmap.draw(songpair.crp_R, xlabel=songpair.song2.filename, ylabel=songpair.song1.filename,
    #                    x_axis='time', y_axis='time')

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
    args = docopt(__doc__)
    medpath, songpath = [args['<medley>'], args['<songdir>']]
    feature = args['--feature']
    lwin = args['--length']
    oti = args['--oti']
    print (f"[feature] {feature}\n[lwin] {lwin}\n")

    import os
    medley = songanal.Song(path=medpath, feature=feature)
    if os.path.isfile(songpath):
        songpairs = [pairanal.SongPair( \
        medley=medley, songpath=songpath, feature=feature,
        lwin=float(lwin), oti=int(oti))]
    else:
        save_data(songpairs)
        songpairs = [pairanal.SongPair( \
            medley=medley, songpath=(songpath+name), feature=feature,
            lwin=float(lwin), oti=int(oti)) \
            for name, oti in load_oti(f"{dir}oti.txt")]

    save_data(songpairs)

if __name__ == '__main__':
    main(sys.argv)
