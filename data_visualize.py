from PIL import Image
import numpy as np


def load_csv(path):
    fin = open(path+'.csv')
    fin.readline()
    return [[int(x) for x in line.strip().split(',')] for line in fin]


def load_npy(path):
    array = np.load(path+'.npy')
    return array


def visual_train(loadfunc):
    img_data = loadfunc('data/train')
    n = 1
    for img in img_data:
        img2d = np.array([255-x for x in img[1:]], dtype=np.uint8).reshape((28, 28))
        imgfile = Image.fromarray(img2d)
        imgfile.save('train_image/{}[{}].png'.format(n, img[0]))
        n += 1
        if n % 1000 == 0:
            print(n)


def visual_test(loadfunc):
    img_data = loadfunc('data/test')
    n = 1
    for img in img_data:
        img2d = np.array([255-x for x in img], dtype=np.uint8).reshape((28, 28))
        imgfile = Image.fromarray(img2d)
        imgfile.save('test_image/{}.png'.format(n))
        n += 1
        if n % 1000 == 0:
            print(n)


if __name__ == '__main__':
    visual_train(load_csv)