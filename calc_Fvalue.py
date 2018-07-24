import pandas as pd
import sys

def get_segment(path):
    data = pd.read_csv(path)

    segstarts = data['segstarts_Q']
    segends = data['segends_Q']

    segments = []
    for start, end in zip(segstarts, segends):
        start_ = int(start.split(",")[0][2:])
        end_ = int(end.split(",")[0][2:])

        segments.append((start_, end_))

    return segments

def calcF(sys, cor, medlen):
    TP = 0.
    FP = 0.
    FN = 0.
    TN = 0.

    for i in range(medlen):
        is_sysseg = (i>=sys[0] and i<=sys[1])
        is_corseg = (i>=cor[0] and i<=cor[1])

        if is_sysseg and is_corseg:
            TP += 1
        if is_sysseg and not is_corseg:
            FP += 1
        if not is_sysseg and is_corseg:
            FN += 1
        if not is_sysseg and not is_corseg:
            TN += 1

    pre = TP / (TP + FP)
    rec = TP / (TP + FN)

    print (TP, FP, FN, TN)

    if TP == 0.0:
        return 0.0
    else:
        return round((2 * rec * pre) / (rec + pre), 4)

def main(path1, path2):
    sysdata = get_segment(path1)
    sysdata = [(0, 430), (431, 1124), (1125, 1266), (1533, 2026), (2027, 2860), (2861, 3333), (3334, 3794)]
    cordata = get_segment(path2)

    medlen = cordata[-1][1]

    Fvals = []
    for sys, col in zip(sysdata, cordata):
        Fvals.append(calcF(sys, col, medlen))

    print (Fvals)




if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
