# coding: utf-8
import librosa.display

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.ticker import Formatter, ScalarFormatter
from matplotlib.ticker import LogLocator, FixedLocator, MaxNLocator
import seaborn as sns

import librosa.core as core


# from librosa
class TimeFormatter(Formatter):
    def __init__(self, lag=False):
        self.lag = lag

    def __call__(self, x, pos=None):
        '''Return the time format as pos'''

        _, dmax = self.axis.get_data_interval()
        vmin, vmax = self.axis.get_view_interval()

        # In lag-time axes, anything greater than dmax / 2 is negative time
        if self.lag and x >= dmax * 0.5:
            # In lag mode, don't tick past the limits of the data
            if x > dmax:
                return ''
            value = np.abs(x - dmax)
            # Do we need to tweak vmin/vmax here?
            sign = '-'
        else:
            value = x
            sign = ''

        if vmax - vmin > 3600:
            s = '{:d}:{:02d}:{:02d}'.format(int(value / 3600.0),
                                            int(np.mod(value / 60.0, 60)),
                                            int(np.mod(value, 60)))
        elif vmax - vmin > 60:
            s = '{:d}:{:02d}'.format(int(value / 60.0),
                                     int(np.mod(value, 60)))
        else:
            s = '{:.2g}'.format(value)

        return '{:s}{:s}'.format(sign, s)


def cmap(data, robust=True, cmap_seq='magma', cmap_bool='gray_r', cmap_div='coolwarm'):
    data = np.atleast_1d(data)

    if data.dtype == 'bool':
        return plt.get_cmap(cmap_bool)

    data = data[np.isfinite(data)]

    if robust:
        min_p, max_p = 2, 98
    else:
        min_p, max_p = 0, 100

    max_val = np.percentile(data, max_p)
    min_val = np.percentile(data, min_p)

    if min_val >= 0 or max_val <= 0:
        return plt.get_cmap(cmap_seq)

    return plt.get_cmap(cmap_div)

def specshow(data, x_coords=None, y_coords=None,
             x_axis=None, y_axis=None,
             sr=22050, hop_length=512,
             fmin=None, fmax=None,
             bins_per_octave=12,
             **kwargs):
    if np.issubdtype(data.dtype, np.complex):
        warnings.warn('Trying to display complex-valued input. '
                      'Showing magnitude instead.')
        data = np.abs(data)

    kwargs.setdefault('cmap', cmap(data))
    kwargs.setdefault('rasterized', True)
    kwargs.setdefault('edgecolors', 'None')
    kwargs.setdefault('shading', 'flat')

    all_params = dict(kwargs=kwargs,
                      sr=sr,
                      fmin=fmin,
                      fmax=fmax,
                      bins_per_octave=bins_per_octave,
                      hop_length=hop_length)

    # Get the x and y coordinates
    y_coords = __mesh_coords(y_axis, y_coords, data.shape[0], **all_params)
    x_coords = __mesh_coords(x_axis, x_coords, data.shape[1], **all_params)

    axes = plt.gca()
    out = axes.pcolormesh(x_coords, y_coords, data, **kwargs)
    plt.sci(out)

    axes.set_xlim(x_coords.min(), x_coords.max())
    axes.set_ylim(y_coords.min(), y_coords.max())

    # Set up axis scaling
    __scale_axes(axes, x_axis, 'x')
    __scale_axes(axes, y_axis, 'y')

    # Construct tickers and locators
    __decorate_axis(axes.xaxis, x_axis)
    __decorate_axis(axes.yaxis, y_axis)

    return axes


def draw(list_, xlabel=None, ylabel=None,
             x_coords=None, y_coords=None,
             x_axis=None, y_axis=None,
             sr=22050, hop_length=512,
             fmin=None, fmax=None,
             bins_per_octave=12,
             **kwargs):

    data = np.array(list_)

    axes = specshow(data, x_axis='time', y_axis='time', cmap='Blues')

    plt.xlabel(xlabel.decode('utf-8'))
    plt.ylabel(ylabel.decode('utf-8'))
    #axes.set_xlabel('[sec]')
    #axes.set_ylabel('[sec]')
    plt.colorbar()

    # set useblit = True on gtkagg for enhanced performance
    cursor = Cursor(axes, useblit=True, color='red', linewidth=1)

    print ("##### heatmap showing...")
    plt.show()

def __mesh_coords(ax_type, coords, n, **kwargs):
    '''Compute axis coordinates'''

    #if coords is not None:
        #if len(coords) < n:
            #raise ParameterError('Coordinate shape mismatch: ' '{}<{}'.format(len(coords), n))
        #    print (len(coords), n)
        #    raise
        #return coords

    coord_map = {'linear': __coord_fft_hz,
                 'hz': __coord_fft_hz,
                 'log': __coord_fft_hz,
                 'mel': __coord_mel_hz,
                 'cqt': __coord_cqt_hz,
                 'cqt_hz': __coord_cqt_hz,
                 'cqt_note': __coord_cqt_hz,
                 'chroma': __coord_chroma,
                 'time': __coord_time,
                 'lag': __coord_time,
                 'tonnetz': __coord_n,
                 'off': __coord_n,
                 'tempo': __coord_tempo,
                 'frames': __coord_n,
                 None: __coord_n}

    if ax_type not in coord_map:
        #raise ParameterError('Unknown axis type: {}'.format(ax_type))
        raise

    return coord_map[ax_type](n, **kwargs)

def __scale_axes(axes, ax_type, which):
    '''Set the axis scaling'''

    kwargs = dict()
    if which == 'x':
        thresh = 'linthreshx'
        base = 'basex'
        scale = 'linscalex'
        scaler = axes.set_xscale
        limit = axes.set_xlim
    else:
        thresh = 'linthreshy'
        base = 'basey'
        scale = 'linscaley'
        scaler = axes.set_yscale
        limit = axes.set_ylim

    # Map ticker scales
    if ax_type == 'mel':
        mode = 'symlog'
        kwargs[thresh] = 1000.0
        kwargs[base] = 2

    elif ax_type == 'log':
        mode = 'symlog'
        kwargs[base] = 2
        kwargs[thresh] = core.note_to_hz('C2')
        kwargs[scale] = 0.5

    elif ax_type in ['cqt', 'cqt_hz', 'cqt_note']:
        mode = 'log'
        kwargs[base] = 2

    elif ax_type == 'tempo':
        mode = 'log'
        kwargs[base] = 2
        limit(16, 480)
    else:
        return

    scaler(mode, **kwargs)

def __decorate_axis(axis, ax_type):
    '''Configure axis tickers, locators, and labels'''

    if ax_type == 'tonnetz':
        axis.set_major_formatter(TonnetzFormatter())
        axis.set_major_locator(FixedLocator(0.5 + np.arange(6)))
        axis.set_label_text('Tonnetz')

    elif ax_type == 'chroma':
        axis.set_major_formatter(ChromaFormatter())
        axis.set_major_locator(FixedLocator(0.5 +
                                            np.add.outer(12 * np.arange(10),
                                                         [0, 2, 4, 5, 7, 9, 11]).ravel()))
        axis.set_label_text('Pitch class')

    elif ax_type == 'tempo':
        axis.set_major_formatter(ScalarFormatter())
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_label_text('BPM')

    elif ax_type == 'time':
        axis.set_major_formatter(TimeFormatter(lag=False))
        axis.set_major_locator(MaxNLocator(prune=None,
                                           steps=[1, 1.5, 5, 6, 10]))
        #axis.set_label_text('Time')

    elif ax_type == 'lag':
        axis.set_major_formatter(TimeFormatter(lag=True))
        axis.set_major_locator(MaxNLocator(prune=None,
                                           steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Lag')

    elif ax_type == 'cqt_note':
        axis.set_major_formatter(NoteFormatter())
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_minor_formatter(NoteFormatter(major=False))
        axis.set_minor_locator(LogLocator(base=2.0,
                                          subs=2.0**(np.arange(1, 12)/12.0)))
        axis.set_label_text('Note')

    elif ax_type in ['cqt_hz']:
        axis.set_major_formatter(LogHzFormatter())
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_minor_formatter(LogHzFormatter(major=False))
        axis.set_minor_locator(LogLocator(base=2.0,
                                          subs=2.0**(np.arange(1, 12)/12.0)))
        axis.set_label_text('Hz')

    elif ax_type in ['mel', 'log']:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_major_locator(SymmetricalLogLocator(axis.get_transform()))
        axis.set_label_text('Hz')

    elif ax_type in ['linear', 'hz']:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_label_text('Hz')

    elif ax_type in ['frames']:
        axis.set_label_text('Frames')

    elif ax_type in ['off', 'none', None]:
        axis.set_label_text('')
        axis.set_ticks([])

def __coord_fft_hz(n, sr=22050, **_kwargs):
    '''Get the frequencies for FFT bins'''
    n_fft = 2 * (n - 1)
    return core.fft_frequencies(sr=sr, n_fft=1+n_fft)

def __coord_mel_hz(n, fmin=0, fmax=11025.0, **_kwargs):
    '''Get the frequencies for Mel bins'''

    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = 11025.0

    return core.mel_frequencies(n+2, fmin=fmin, fmax=fmax)[1:]

def __coord_cqt_hz(n, fmin=None, bins_per_octave=12, **_kwargs):
    '''Get CQT bin frequencies'''
    if fmin is None:
        fmin = core.note_to_hz('C1')
    return core.cqt_frequencies(n+1, fmin=fmin, bins_per_octave=bins_per_octave)

def __coord_chroma(n, bins_per_octave=12, **_kwargs):
    '''Get chroma bin numbers'''
    return np.linspace(0, (12.0 * n) / bins_per_octave, num=n+1, endpoint=True)

def __coord_tempo(n, sr=22050, hop_length=512, **_kwargs):
    '''Tempo coordinates'''
    return core.tempo_frequencies(n+1, sr=sr, hop_length=hop_length)

def __coord_n(n, **_kwargs):
    '''Get bare positions'''
    return np.arange(n+1)

def __coord_time(n, sr=22050, hop_length=512, **_kwargs):
    '''Get time coordinates from frames'''
    return core.frames_to_time(np.arange(n+1), sr=sr, hop_length=hop_length)
