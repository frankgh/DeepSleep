import glob
import os

import numpy as np
import pyedflib


def process_hypnogram_file(path):
    f = pyedflib.EdfReader(path)
    annotations = f.readAnnotations()
    f._close()
    return annotations


def process_psg_file(path):
    """

    :param path: PSG edf file path
    :return: DataFrame
    """
    f = pyedflib.EdfReader(path)
    n = f.signals_in_file
    freq = f.getSampleFrequencies()
    max_freq = freq.max()
    signal_labels = f.getSignalLabels()
    sigbufs = np.full((n, f.getNSamples()[0]), np.nan)
    for i in xrange(n):
        signal = f.readSignal(i)
        signal_ratio = int(max_freq / freq[i])
        sigbufs[i, ::signal_ratio] = signal
    f._close()
    return sigbufs, signal_labels, max_freq


class SleepDataParser(object):
    def __init__(self,
                 data_dir,
                 output_dir):
        super(SleepDataParser, self).__init__()
        self.data_dir = data_dir
        self.output_dir = output_dir

    def parse(self):
        psg_files = sorted(glob.glob(os.path.join(self.data_dir, '*-PSG.edf')))

        # data = np.load('/Users/francisco/src/sdvt/models/kerasvis-afg/patients/p0.npz')

        # p0X = data['X']
        # p0Y = data['Y']

        # shape_format = p0X.shape

        # with open('/Users/francisco/src/DeepSleep/mike.csv', 'wb') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=',')

        # for k in xrange(p0X.shape[0]):
        #     for l in xrange(6000, 9000):
        #         writer.writerow([p0X[k, l, 0], p0X[k, l, 1], p0X[k, l, 2]])

        for psg_file in psg_files:
            # Find the corresponding hyp_file
            hyp_file = glob.glob(psg_file[:-9] + '*-Hypnogram.edf')[0]
            signals, signal_labels, max_freq = process_psg_file(psg_file)
            annotations = process_hypnogram_file(hyp_file)
            X, Y = [], []

            # frame = pd.DataFrame(signals[0:90000])
            # frame.to_csv('/Users/francisco/src/DeepSleep/me.csv', sep=',', encoding='utf-8')

            # frame1 = pd.DataFrame(p0X)
            # frame1.to_csv('/Users/francisco/src/DeepSleep/mike.csv', sep=',', encoding='utf-8')

            count = 0

            # with open('/Users/francisco/src/DeepSleep/mine.csv', 'wb') as csvfile:
            for i in range(len(annotations[0])):
                label = annotations[2][i]
                start, size = int(annotations[0][i] * max_freq), int(annotations[1][i] * max_freq)
                if label == 'Sleep stage W' or size % 3000 != 0:
                    continue

                    # writer = csv.writer(csvfile, delimiter=',')
                    #
                    # for k in xrange(start, start + size):
                    #     writer.writerow([signals[0, k], signals[1, k], signals[2, k], signals[3, k]])

                for j in xrange(size // 3000):
                    s = start + j * 3000
                    e = s + 3000

                    X.append(signals[0:3, s:e])
                    Y.append(label)

                    # print signals[0, s:s + 10]  # , s, e, label
                    count += 1


def test(args=None):
    import sys
    if not args:
        args = sys.argv[1:]
    if not args:
        print 'usage: %s <number-of-random-items | item item item ...>' % \
              sys.argv[0]
        sys.exit()

    data_dir = args[0]
    output_dir = args[1]

    SleepDataParser(data_dir, output_dir).parse()


if __name__ == '__main__':
    test(args=['/Users/francisco/src/SleepData', '/Users/francisco/src/SleepData/output'])
