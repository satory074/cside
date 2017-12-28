# coding: utf-8
import librosa
import numpy as np
import os
import scipy.signal

import melody_extraction as meloext

def locmax(vec, indices=False):
    """
    Return a boolean vector of which points in vec are local maxima.
    End points are peaks if larger than single neighbors.
    if indices=True, return the indices of the True values instead of the boolean vector.
    """
    nbr = np.zeros(len(vec)+1, dtype=bool)
    nbr[0] = True
    nbr[1:-1] = np.greater_equal(vec[1:], vec[-1])
    maxmask = (nbr[:-1] & ~nbr[1:])

    if indices:
        return np.nonzero(maxmask)[0]
    else:
        return maxmask

# DENSITY controls the density of landmarks found (approx DENSITY per sec)
DENSITY = 20.0
# OVERSAMP > 1 tries to generate extra landmarks by decaying faster
OVERSAMP = 1
N_FFT = 512
N_HOP = 256
# spectrogram enhancement
HPF_POLE = 0.98

class Song:
    # optimaization: cache pre-calculated Gaussian profile
    __sp_width = None
    __sp_len = None
    __sp_vals = []

    def __init__(self, path, feature, is_extract, is_decompose):
        self.density = DENSITY
        self.n_fft = N_FFT
        self.n_hop = N_HOP
        #how wide to spread peaks
        self.f_sd = 30.0
        # Maximum number of local maxima to keep per frame
        self.maxpksperframe = 50

        self.path = path
        self.feature = feature
        self.filename = (os.path.splitext(path)[0]).split("/")[-1]
        self.h, self.g = self._path2g(feature, is_extract, is_decompose)
        self.htr = self.h

    def spreadpeaksinvector(self, vector, width=4.0):
        """
        Create a blurred version of vector, where each of the local maxes
        is spread by a gaussian with SD <width>.
        """
        npts = len(vector)
        peaks = locmax(vector, indices=True)
        return self.spreadpeaks(zip(peaks, vector[peaks]),
                                npoints=npts, width=width)

    def spreadpeaks(self, peaks, npoints=None, width=4.0, base=None):
        """
        Generate a vector consisting of the max of a set of Gaussian bumps
        :params:
          peaks : list
            list of (index, value) pairs giving the center point and height
            of each gaussian
          npoints : int
            the length of the output vector (needed if base not provided)
          width : float
            the half-width of the Gaussians to lay down at each point
          base : np.array
            optional initial lower bound to place Gaussians above
        :returns:
          vector : np.array(npoints)
            the maximum across all the scaled Gaussians
        """
        if base is None:
            vec = np.zeros(npoints)
        else:
            npoints = len(base)
            vec = np.copy(base)

        if width != self.__sp_width or npoints != self.__sp_len:
            # Need to calculate new vector
            self.__sp_width = width
            self.__sp_len = npoints
            self.__sp_vals = np.exp(-0.5*((np.arange(-npoints, npoints+1)
                                    / float(width))**2))

        # Now the actual function
        for pos, val in peaks:
            vec = np.maximum(vec, val*self.__sp_vals[np.arange(npoints)
                                                    + npoints - pos])

        return vec

    def _decaying_threshold_fwd_prune(self, sgram, a_dec):
        """
        forward pass of findpeaks
        initial threshold envelope based on peaks in first 10 frames
        """
        (srows, scols) = np.shape(sgram)
        sthresh = self.spreadpeaksinvector(
            np.max(sgram[:, :np.minimum(10, scols)], axis=1), self.f_sd
        )

        #peaks = np.zeros((srows, scols))
        peaks = sgram / 5.0

        # optimization of mask update
        __sp_pts = len(sthresh)
        __sp_v = self.__sp_vals

        for col in range(scols):
            s_col = sgram[:, col]
            # Find local magnitude peaks that are above threshold
            sdmaxposs = np.nonzero(locmax(s_col) * (s_col > sthresh))[0]

            # Work down list of peaks in order of their absolute value
            # above threshold
            valspeaks = sorted(zip(s_col[sdmaxposs], sdmaxposs), reverse=True)
            for val, peakpos in valspeaks[:self.maxpksperframe]:
                # Optimization - inline the core function within spreadpeaks
                sthresh = np.maximum(sthresh,
                                    val*__sp_v[(__sp_pts - peakpos):
                                                (2*__sp_pts - peakpos)])
                #peaks[peakpos, col] = 1
                peaks[peakpos, col] = sgram[peakpos, col]
            sthresh *= a_dec

        return peaks

    def _decaying_threshold_bwd_prune_peaks(self, sgram, peaks, a_dec):
        """backwards pass of findpeaks"""
        scols = np.shape(sgram)[1]
        # Backwards filter to prune peaks
        sthresh = self.spreadpeaksinvector(sgram[:, -1], self.f_sd)

        for col in range(scols, 0, -1):
            pkposs = np.nonzero(peaks[:, col-1])[0]
            peakvals = sgram[pkposs, col-1]
            for val, peakpos in sorted(zip(peakvals, pkposs), reverse=True):
                if val >= sthresh[peakpos]:
                    # Setup the threshold
                    sthresh = self.spreadpeaks([(peakpos, val)], base=sthresh,
                                                width=self.f_sd)
                    # Delete any following peak (threshold, but be sure)
                    if col < scols:
                        #peaks[peakpos, col] = 0
                        peaks[peakpos, col] = peaks[peakpos, col] / 5.0
                else:
                    # delete the peak
                    # peaks[peakpos, col-1] = 0
                    peaks[peakpos, col-1] = peaks[peakpos, col-1] / 5.0
            sthresh = a_dec * sthresh
        return peaks

    def find_peaks(self, d, sr):
        """Find the local peaks in the spectrogram as basis for fingerprints.
         Returns a list of (time_frame, freq_bin) pairs.

         :params:
           d - np.array of float
             Input waveform as 1D vector

           sr - int
             Sampling rate of d (not used)

         :returns:
           peaklist - list of (int, int)
             Ordered list of landmark peaks found in STFT. First value of each pair
             is the time index (in STFT frames, i.e., units of n_hop/sr secs),
             second is the FFT bin (in units of sr/n_fft Hz).
         """
        if len(d)==0:
             return []

        # masking envelope decay constant (???Magical number???)
        a_dec = (1.0 - 0.01*(self.density*np.sqrt(self.n_hop/352.8)/35.0)) \
                **(1.0/OVERSAMP)

        # Take spectrogram
        mywin = np.hanning(self.n_fft+2)[1:-1]
        #sgram = np.abs(librosa.stft(d, n_fft=self.n_fft,
        #                            hop_length=self.n_hop,
        #                            window=mywin))
        sgram = librosa.feature.chroma_cqt(y=d, sr=sr)
        sgrammax = np.max(sgram)
        if sgrammax > 0.0:
            sgram = np.log(np.maximum(sgram, np.max(sgram)/1e6))
            sgram = sgram - np.mean(sgram)
        else:
            # The sgram is identically zero, i.e., the input signal was identically
            # zero. Not good, but let's let it through for now.
            print ("find_peaks: Warning: input signak is identically zero.")

        # High-pass filter onset emphasis
        # [:-1, ] discards top bin (nyquist) of sgram so bins fit in 8 bits
        sgram = np.array([scipy.signal.lfilter([1, -1],
                                                [1, -(HPF_POLE)**\
                                                (1/OVERSAMP)], s_row)
                        for s_row in sgram])[:-1,]

        # Prune to keep only local maxima in spectrum that appear above an online,
        # decaying threshold
        peaks = self._decaying_threshold_fwd_prune(sgram, a_dec)
        # Further prune these peaks working backwards in time, to remove small peaks
        # that are closely followed by a large peak
        peaks = self._decaying_threshold_bwd_prune_peaks(sgram, peaks, a_dec)

        # build a list of peaks we ended up with
        scols = np.shape(sgram)[1]
        pklist = []
        for col in xrange(scols):
            for bin_ in np.nonzero(peaks[:, col])[0]:
                pklist.append((col, bin_))

        #return pklist
        return peaks

    def wavfile2peaks(self, y, sr):
        # Calculate hashes with optional part-frame shifts
        peaklists = []
        return self.find_peaks(y, sr)

    def _path2g(self, feature, is_extract, is_decompose):
        print ("### " + self.filename + " load...")

        # extract chroma
        y, sr = librosa.load(self.path)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)

        ha_sum = np.sum(chroma_cqt, axis=1)
        g = ha_sum / np.max(ha_sum)

        if is_extract:
            h = meloext.extract(y, sr, is_decompose)
            return h, g

        if feature == 'chroma':
            h = chroma_cqt
        if feature == 'cqt':
            h = librosa.cqt(y, sr=sr)
        if feature == 'mfcc':
            h = librosa.feature.mfcc(y=y, sr=sr)
        if feature == 'peak':
            h = self.wavfile2peaks(y, sr)

        return h, g
