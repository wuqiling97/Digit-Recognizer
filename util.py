import time
import numpy as np
import random


def current_time():
    return time.strftime('%H:%M:%S')


def csv2npy(filename):
    fin = open(filename)
    next(fin)
    alldata = np.array([[int(num) for num in line.strip().split(',')] for line in fin], dtype=np.float32)
    np.save(filename.replace('.csv', '.npy'), alldata)
    print('{} saved'.format(filename))


class DataSet:
    def __init__(self, filename):
        """filename: .npy file"""
        assert filename.endswith('.npy')

        def dense2onehot(num, num_classes=10):
            arr = np.zeros(num_classes, dtype=np.float32)
            arr[int(num)] = 1
            return arr

        alldata = np.load(filename)
        self.datasize = len(alldata)
        if 'train' in filename:
            self.labels = np.array([dense2onehot(label) for label in alldata[..., 0]])
            self.images = alldata[..., 1:] / 256
        elif 'test' in filename:
            self.labels = None
            self.images = alldata / 256
        else:
            raise NameError('Please put "test" or "train" in filename')

    def next_batch(self, size):
        labels = []
        images = []
        for i in random.sample(range(self.datasize), size):
            labels.append(self.labels[i])
            images.append(self.images[i])
        return np.array(images), np.array(labels)

    def testbatches(self, size):
        idx = 0
        while idx < self.datasize:
            r = idx + min(size, self.datasize-idx)
            yield self.images[idx:r]
            idx = r


if __name__ == '__main__':
    csv2npy('data/train.csv')
    csv2npy('data/test.csv')
    # train = DataSet('data/train.npy')
