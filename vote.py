import os
import numpy as np
from PIL import Image


def write_result(res):
    """res: list [label1, label2, ...]"""
    fout = open('test_ans.csv', 'w')
    fout.write('ImageId,Label\n')
    for i, label in enumerate(res):
        fout.write('{},{}\n'.format(i+1, label))
    fout.close()


def openimg(path):
    img = Image.open(path)
    img.show()


anslst = []
votelst = []
for fname in os.listdir('.'):
    if fname.endswith('.csv') and fname.startswith('gyc'):
        fin = open(fname)
        next(fin)
        lst = [int(line.strip().split(',')[1]) for line in fin]
        anslst.append(lst)
        print(fname)

for i, ans in enumerate(zip(*anslst)):
    if not all(x == y for x, y in zip(ans, ans[1:])):
        lst = [0]*10
        for j in ans:
            lst[j] += 1
        index = np.argmax(lst)
        if lst[index] <= len(ans)/2:
            print('image {} unequal, ans: {}'.format(i+1, ans))
            # openimg('test_image/{}.png'.format(i+1))
            # num = input('input your answer\n')
            votelst.append(ans[0])
        else:
            votelst.append(index)
            print('image {} voted, ans: {}'.format(i+1, ans))
    else:
        votelst.append(ans[0])

write_result(votelst)

