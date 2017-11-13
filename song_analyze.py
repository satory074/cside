import os
import numpy as np
import librosa

# Not maintenance yet
def librosaRP(ga):
    import matplotlib.pyplot as plt
    R = librosa.segment.recurrence_matrix(ga, metric='cosine')
    R_aff = librosa.segment.recurrence_matrix(ga, mode='affinity')

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    librosa.display.specshow(R, x_axis='time', y_axis='time')
    plt.title('Binary recurrence (symmetric)')
    plt.subplot(1, 2, 2)
    librosa.display.specshow(R_aff, x_axis='time', y_axis='time', cmap='magma_r')
    plt.title('Affinity recurrence')
    plt.tight_layout()
    plt.show()

class Song:
    def __init__(self, is_recalc, path):
        # savepath ="/Users/satory43/Desktop/Programs/Python/py2/csi/data/song/"
        savepath = "/Volumes/satoriHD/data/song/"

        self.path = path
        self.filename = (os.path.splitext(path)[0]).split("/")[-1]
        self.ifp = True if is_recalc else self._is_first_processing(savepath)
        self.h, self.g = self._path2g(savepath)
        self.htr = self.h

    def _is_first_processing(self, savepath):
        for x in os.listdir(savepath + "chroma/"):
            if os.path.isfile(savepath + "chroma/" + x):
                if x.split(".")[0] == self.filename:
                    return False

        return True

    def wavfile2peaks(self, filename)

    def _path2g(self, savepath):
        print ("##### " + self.filename + " load...")

        if self.ifp:
            # extract chroma
            y, sr = librosa.load(self.path)
            chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

            ha_sum = np.sum(chroma_cq, axis=1)
            g = ha_sum / np.max(ha_sum)

            wavfile2peaks(self.path)

            # output
            np.savetxt(savepath + "chroma/" + self.filename + ".csv", chroma_cq, delimiter=',')
            np.savetxt(savepath + "exchroma/" + self.filename + ".csv", g, delimiter=',')

            return chroma_cq, g
        else:
            # finding same file
            chroma_cq = np.loadtxt(savepath + "chroma/" + self.filename + ".csv", delimiter=',')
            g = np.loadtxt(savepath + "exchroma/" + self.filename + ".csv", delimiter=',')

            return chroma_cq, g
