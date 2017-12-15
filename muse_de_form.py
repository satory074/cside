"""
Music Segments Detector for Medley

usage: muse_de_form.py (<path1> <path2>) [options]

options:
    -f <feature>, --feature <feature>    Used for features that generate CRP [default: chroma]
    -m [mat], --matrices [mat]           List of matrices to be calculated. [default: q]
    -r <rate>, --rate <rate>             How much rate of CRP plots [default: 0.1]
    --help                               Show this help message and exit
"""

from docopt import docopt
from scipy import stats
import librosa.display
import numpy as np
import os
import sys

import draw_heatmap
import songpair_analyze


def main(argv):
    args = docopt(__doc__)

    paths = [args['<path1>'], args['<path2>']]

    feature = args['--feature']
    flist = ['chroma', 'cqt', 'mfcc', 'peak']
    if feature not in flist:
        print ("Undefined feature.")
        exit()
    print ("[ feature ] " + feature )

    cpr = float(args['--rate'])
    print ("[ cpr ] " + str(cpr) )

    matrices = list(args['--matrices'])
    mlist = ['l', 'q', 's']
    for mat in matrices:
        if mat not in mlist:
            print ("Undefined matrix.")
            exit()
    print ("[ calc matrices ] " + str(matrices))
    print

    songpair = songpair_analyze.SongPair(paths, feature, cpr, matrices)

    draw_heatmap.draw(songpair.crp_R, xlabel=songpair.song2.filename, ylabel=songpair.song1.filename,
                        x_axis='time', y_axis='time')
    #songpair.draw_heatmap(songpair.crp_L, song1.filename, song2.filename)
    #songpair.draw_heatmap(songpair.crp_S, song1.filename, song2.filename)
    #songpair.draw_heatmap(songpair.crp_Q, song1.filename, song2.filename)


if __name__ == '__main__':
    main(sys.argv)
