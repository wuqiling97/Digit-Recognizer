from PIL import Image
import numpy as np


def visual_train():
    train = open('train.csv')
    train.readline()
    img_data = [line.strip().split(',') for line in train]
    n = 1
    for img in img_data:
        img2d = np.array([255-int(x) for x in img[1:]], dtype=np.uint8).reshape((28, 28))
        imgfile = Image.fromarray(img2d)
        imgfile.save('train_image/{}[{}].png'.format(n, img[0]))
        n += 1
        if n % 1000 == 0:
            print(n)


def visual_test():
    test = open('test.csv')
    test.readline()
    img_data = [line.strip().split(',') for line in test]
    n = 1
    for img in img_data:
        img2d = np.array([255-int(x) for x in img], dtype=np.uint8).reshape((28, 28))
        imgfile = Image.fromarray(img2d)
        imgfile.save('test_image/{}.png'.format(n))
        n += 1
        if n % 1000 == 0:
            print(n)


visual_test()