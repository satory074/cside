#coding: utf-8
"""
Music Segments Detector for Medley

usage: muse_de_form.py (<path1> <path2>) [options]

options:
    -d, --decompose                      Decompose an audio time series into harmonic and percussive components.
    -e, --extract                        Do melody extract
    -f <feature>, --feature <feature>    Used for features that generate CRP [default: cqt]
    -o <oti>, --oti <oti>
    --help                               Show this help message and exit
"""

from docopt import docopt
import sys

import draw_heatmap
import songpair_analyze

def main(argv):
    args = docopt(__doc__)

    # path
    paths = [args['<path1>'], args['<path2>']]

    # feature
    feature = args['--feature']
    flist = ['chroma', 'cqt']
    if feature not in flist:
        print ("Undefined feature.")
        exit()
    print ("[ feature ] " + feature )

    # OTI
    oti = int(args['--oti'])

    songpair = songpair_analyze.SongPair(paths, feature, oti)

    #draw_heatmap.draw(songpair.crp_R, xlabel=songpair.song2.filename, ylabel=songpair.song1.filename,
    #                    x_axis='time', y_axis='time')

    output = [songpair.song1.filename, len(songpair.crp_R), songpair.song2.filename, len(songpair.crp_R[0]),
            songpair.oti, feature,
            songpair.Qmax, songpair.segends_Q]
    f = open("output/segdata/" + songpair.filename + '.txt', 'w')
    for x in output:
        f.write(str(x) + "\n")
    f.close()

if __name__ == '__main__':
    main(sys.argv)
