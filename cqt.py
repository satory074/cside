#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spectral feature extraction"""

import librosa

import numpy as np
import scipy
import scipy.signal
import scipy.fftpack


def chroma_cqt(y=None, sr=22050, C=None, hop_length=512, fmin=0, fmax=83,
               norm=np.inf, threshold=0.0, tuning=None, n_chroma=12,
               n_octaves=7, window=None, bins_per_octave=None, tempo=None,  feature='cqt'):
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
    print (spb)
    spb = int(spb)

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

    if feature == 'chroma':
        # Map to chroma
        cq_to_chr = librosa.filters.cq_to_chroma(C.shape[0],
                                         bins_per_octave=bins_per_octave,
                                         n_chroma=n_chroma,
                                         window=window)
        chroma = cq_to_chr.dot(C)
    #chroma = C

    # threshold
    nplots = int(chroma.shape[1] / (spb / 4.0))
    print (nplots)

    sortedlist = np.sort(chroma.reshape(-1,))[::-1]
    threshold = sortedlist[nplots]
    #threshold = 0.0

    chroma[chroma < threshold] = 0.0

    # Normalize
    if norm is not None:
        chroma = librosa.util.normalize(chroma, norm=norm, axis=0)

    return chroma



def chroma_cens(y=None, sr=22050, C=None, hop_length=512, fmin=None,
                tuning=None, n_chroma=12,
                n_octaves=7, bins_per_octave=None, cqt_mode='full', window=None,
                norm=2, win_len_smooth=41):
    r'''Computes the chroma variant "Chroma Energy Normalized" (CENS), following [1]_.

    .. [1] Meinard Müller and Sebastian Ewert
           Chroma Toolbox: MATLAB implementations for extracting variants of chroma-based audio features
           In Proceedings of the International Conference on Music Information Retrieval (ISMIR), 2011.

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

    fmin : float > 0
        minimum frequency to analyze in the CQT.
        Default: 'C1' ~= 32.7 Hz

    norm : int > 0, +-np.inf, or None
        Column-wise normalization of the chromagram.

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

    cqt_mode : ['full', 'hybrid']
        Constant-Q transform mode

    win_len_smooth : int > 0
        Length of temporal smoothing window.
        Default: 41

    Returns
    -------
    chroma_cens : np.ndarray [shape=(n_chroma, t)]
        The output cens-chromagram

    See Also
    --------
    chroma_cqt
        Compute a chromagram from a constant-Q transform.

    chroma_stft
        Compute a chromagram from an STFT spectrogram or waveform.

    Examples
    --------
    Compare standard cqt chroma to CENS.


    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      offset=10, duration=15)
    >>> chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    >>> chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2,1,1)
    >>> librosa.display.specshow(chroma_cq, y_axis='chroma')
    >>> plt.title('chroma_cq')
    >>> plt.colorbar()
    >>> plt.subplot(2,1,2)
    >>> librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time')
    >>> plt.title('chroma_cens')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    '''
    chroma = chroma_cqt(y=y, C=C, sr=sr,
                        hop_length=hop_length,
                        fmin=fmin,
                        bins_per_octave=bins_per_octave,
                        tuning=tuning,
                        norm=None,
                        n_chroma=n_chroma,
                        n_octaves=n_octaves,
                        cqt_mode=cqt_mode,
                        window=window)

    # L1-Normalization
    chroma = util.normalize(chroma, norm=1, axis=0)

    # Quantize amplitudes
    QUANT_STEPS = [0.4, 0.2, 0.1, 0.05]
    QUANT_WEIGHTS = [0.25, 0.25, 0.25, 0.25]

    chroma_quant = np.zeros_like(chroma)

    for cur_quant_step_idx, cur_quant_step in enumerate(QUANT_STEPS):
        chroma_quant += (chroma > cur_quant_step) * QUANT_WEIGHTS[cur_quant_step_idx]

    # Apply temporal smoothing
    win = filters.get_window('hann', win_len_smooth + 2, fftbins=False)
    win /= np.sum(win)
    win = np.atleast_2d(win)

    cens = scipy.signal.convolve2d(chroma_quant, win,
                                   mode='same', boundary='fill')

    # L2-Normalization
    return util.normalize(cens, norm=norm, axis=0)
