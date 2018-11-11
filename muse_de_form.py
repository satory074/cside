"""
Music Segments Detector for Medley

Usage:
    muse_de_form.py <dir_med> [options]
    muse_de_form.py (<path_med> <path_song>) [options]

Options:
    -f <feature>, --feature <feature>    Used for features that generate CRP [default: cqt]
    -l <len>, --length <len>             number of half notes [default: 0.5]
    --help                               Show this help message and exit
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docopt import docopt
from song_analyze import Song
from songpair_analyze import SongPair


def load_data(path):
    with open(path) as f:
        li = [tuple(d.split(', ')) for d in f.readlines()]

    return li


def save_data(songpairs):
    data = pd.DataFrame(
        [], columns=['Name', 'length', 'Name', 'length','Qmax',]
    )
    d = [str(i) for i in np.arange(len(songpairs[0].Qmaxlist))]
    qmax = []
    index = []
    id = 0
    for songpair in songpairs:
        id += 1
        data.loc[id] = [songpair.medley.name,
                        songpair.Q.shape[0],
                        songpair.song.name,
                        songpair.Q.shape[1],
                        # songpair.oti,
                        # songpair.feature,
                        songpair.Q.max(),
                        # songpair.segends_Q
                        ]
        qmax.append(songpair.Qmaxlist)
        index.append(songpair.song.name)

    data.to_csv(f"output/intensive/{songpair.medley.name}.csv")
    pd.DataFrame(qmax, index=index).to_csv(f"output/Qmaxlist/{songpair.medley.name}")

def main(argv):
    ### args ###
    args = docopt(__doc__)
    dir_med = args['<dir_med>']
    path_med, path_song = [args['<path_med>'], args['<path_song>']]
    feature = args['--feature']
    lwin = args['--length']

    print(f"[feature] {feature}")
    print(f"[lwin] {lwin}")
    print()

    ###  ###
    if dir_med:
        li_song = [Song(path=f"{dir_med}/{name}", feat=feature)
                   for name, oti in load_data(f"{dir_med}/data.txt")]
        li_songpair = [SongPair(med=li_song[0], song=song, lwin=lwin)
                       for song in li_song[1:]]
    else:
        med = Song(path=path_med, feat=feature)
        song = Song(path=path_song, feat=feature)
        li_songpair = [SongPair(med=med, song=song, lwin=lwin)]

    for sp in li_songpair:
        plt.plot(np.arange(sp.song1.len_), sp.Qmaxlist, label=sp.song2.name)
        plt.legend()
    plt.show()

    #save_data(li_songpair)

if __name__ == '__main__':
    main(sys.argv)
