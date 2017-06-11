import time
import numpy as np
import random
import os


def current_time():
    return time.strftime('%H:%M:%S')


def csv2npy(filename):
    fin = open(filename)
    next(fin)
    alldata = np.array([[int(num) for num in line.strip().split(',')] for line in fin], dtype=np.float32)
    np.save(filename.replace('.csv', '.npy'), alldata)
    print('{} saved'.format(filename))


class DataSet:
    def __init__(self, data_arr, haslabel):
        def dense2onehot(num, num_classes=10):
            arr = np.zeros(num_classes, dtype=np.float32)
            arr[int(num)] = 1
            return arr

        self.datasize = len(data_arr)
        if haslabel:
            self.labels = np.array([dense2onehot(label) for label in data_arr[..., 0]], copy=False)
            self.images = data_arr[:, 1:] / 256
        else:
            self.labels = None
            self.images = data_arr / 256

    def next_batch(self, size):
        labels = []
        images = []
        for i in random.sample(range(self.datasize), size):
            labels.append(self.labels[i])
            images.append(self.images[i])
        return np.array(images, copy=False), np.array(labels, copy=False)

    def testbatches(self, size):
        idx = 0
        while idx < self.datasize:
            r = idx + min(size, self.datasize-idx)
            yield self.images[idx:r]
            idx = r


class DataCollection:
    def __init__(self, folder, vali_size):
        traindir = os.path.join(folder, 'train.npy')
        testdir = os.path.join(folder, 'test.npy')

        test = np.load(testdir)
        self.test = DataSet(test, False)
        train = np.load(traindir)
        vali = None
        if vali_size > 0:
            vali = train[-vali_size:]
            train = train[:-vali_size]
            self.validation = DataSet(vali, True)
            self.train = DataSet(train, True)
        else:
            self.train = DataSet(train, True)
            self.validation = None


if __name__ == '__main__':
    csv2npy('data/train.csv')
    csv2npy('data/test.csv')
