import time
import numpy as np
import random
import os


def current_time():
    return time.strftime('%H:%M:%S')


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
            r = idx + min(size, self.datasize - idx)
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


def write_result(res, fname):
    """res: list [label1, label2, ...]"""
    fout = open(fname, 'w')
    fout.write('ImageId,Label\n')
    for i, label in enumerate(res):
        fout.write('{},{}\n'.format(i+1, label))
    fout.close()


def data_augment(alldata):
    ret = []
    for i, data in enumerate(alldata):
        ret.append(data)
        rawimg = data[1:]
        label = data[0:1]
        image = rawimg.reshape((28, 28))
        cnt = 0
        while True:
            step = random.choice((-1, -2, 1, 2))
            axis = random.randint(0, 1)
            newimg = np.roll(image, step, axis)
            slice = None
            if step > 0:
                if axis == 0:
                    slice = image[-step:, :]
                else:
                    slice = image[:, -step:]
            else:
                if axis == 0:
                    slice = image[:-step, :]
                else:
                    slice = image[:, :-step]
            # ensure slice == 0
            if not np.any(slice):
                break
            cnt += 1
            if cnt >= 10:
                print('cannot find a suitable step, step={}, axis={}, num={}'.format(step, axis, i))
                print(image, slice, sep='\n')
                input()

        ret.append(np.concatenate((label, newimg.reshape(-1))))
    return ret


def csv2npy(filename, is_augment):
    fin = open(filename)
    next(fin)
    alldata = [np.array([int(num) for num in line.strip().split(',')]) for line in fin]
    if is_augment:
        alldata = data_augment(alldata)
        random.shuffle(alldata)
    array = np.array(alldata, copy=False)
    np.save(filename.replace('.csv', '.npy'), array)
    print('{} saved'.format(filename))


# fin = open('data/train.csv')
# next(fin)
# alldata = [np.array([int(num) for num in line.strip().split(',')]) for line in fin]
# print('alldata gen')
# alldata = data_augment(alldata)


if __name__ == '__main__':
    csv2npy('data/train.csv', True)
    csv2npy('data/test.csv', False)
