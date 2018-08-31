"""Spectral feature extraction"""

import librosa
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def chroma_cqt(y=None, sr=22050, C=None, hop_length=512, fmin=0, fmax=83,
               norm=np.inf, threshold=0.0, tuning=None, n_chroma=12,
               n_octaves=7, window=None, bins_per_octave=None, tempo=1.,  feature='cqt'):
    r'''Constant-Q chromagram

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : number > 0
        sampling rate of `y`

    C : np.ndarray [shape=(d, t)] [Optional]
        a pre-computed constant-Q spectrogram

    hop_length : int > 0
        number of samples between successive chroma frames

    fmin : 0 <= int < fmax
        If the frequency is less than fmin, the value in this range is 0.0.

    fmax : fmin < int <= 83
        If the frequency is more than fmax, the value in this range is 0.0.

    norm : int > 0, +-np.inf, or None
        Column-wise normalization of the chromagram.

    threshold : float
        Pre-normalization energy threshold.  Values below the
        threshold are discarded, resulting in a sparse chromagram.

    tuning : float
        Deviation (in cents) from A440 tuning

    n_chroma : int > 0
        Number of chroma bins to produce

    n_octaves : int > 0
        Number of octaves to analyze above `fmin`

    window : None or np.ndarray
        Optional window parameter to `filters.cq_to_chroma`

    bins_per_octave : int > 0
        Number of bins per octave in the CQT.
        Default: matches `n_chroma`

    tempo : float
        hoge

    feature : chroma or cqt
        hoge

    Returns
    -------
    chromagram : np.ndarray [shape=(n_chroma, t)]
        The output chromagram

    '''
    spb = ((60 / tempo) * (sr / float(hop_length))) * 0.5 #samples per beat

    if bins_per_octave is None:
        bins_per_octave = n_chroma

    # Build the CQT if we don't have one already
    if C is None:
        C = np.abs(librosa.core.cqt(y, sr=sr,
                                      hop_length=hop_length,
                                      fmin=None,
                                      n_bins=n_octaves * bins_per_octave,
                                      bins_per_octave=bins_per_octave,
                                      tuning=tuning))

    # fmin, fmax
    C[:fmin] = 0.0
    C[fmax:] = 0.0

    chroma = C
    if feature == 'chroma':
        # Map to chroma
        cq_to_chr = librosa.filters.cq_to_chroma(C.shape[0],
                                         bins_per_octave=bins_per_octave,
                                         n_chroma=n_chroma,
                                         window=window)
        chroma = cq_to_chr.dot(C)

    # threshold
    nplots = int(chroma.shape[1] / (spb / 2.0))

    sortedlist = np.sort(chroma.reshape(-1,))[::-1]
    #threshold = sortedlist[nplots]
    threshold = 0.0

    chroma[chroma < threshold] = 0.0

    # Normalize
    if norm is not None:
        chroma = librosa.util.normalize(chroma, norm=norm, axis=0)

    return chroma
