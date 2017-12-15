import vamp
import librosa
#import essentia.standard as es
import matplotlib.pyplot as plt
#%matplotlib inline
#from __future__ import print_function
import sys

def main(path):
    # This is how we load audio using Librosa
    audio, sr = librosa.load(path, sr=44100, mono=True)
    params = {"minfqr": 50.0, "maxfqr": 1800.0, "voicing":0.2, "minpeaksalience": 0}
    data = vamp.collect(audio, sr, "mtg-melodia:melodia")

    # vector is a tuple of two values: the hop size used for analysis and the array of pitch values
    # Note that the hop size is *always* equal to 128/44100.0 = 2.9 ms
    hop, melody = data['vector']

    import numpy as np
    timestamps = 8 * 128/44100.0 + np.arange(len(melody)) * (128/44100.0)

    # Melodia returns unvoiced (=no melody) sections as negative values. So by default, we get:
    plt.figure(figsize=(18,6))
    plt.plot(timestamps, melody)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1])
