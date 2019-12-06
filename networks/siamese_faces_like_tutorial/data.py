import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image


size = (46, 56)
repeat_s = 25
repeat_f = 25

dir = 'data/orl_faces/'


def cheaker(res, target):
    for i in range(res.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(res[i, 0])
        ax1 = fig.add_subplot(122)
        ax1.imshow(res[i, 1])
        print(target[i])
        plt.show()


def prepare(dir):
    def same(f):
        same = []
        for aa in range(repeat_s):
            for i in f:
                files = []
                for root, dirs, files in os.walk(i):
                    pass
                for t in range(len(files)):
                    img0 = plt.imread(f'{i}/{files[t]}')
                    img0 = cv2.resize(img0, dsize=size)
                    img0 = np.reshape(img0, (1, img0.shape[0], img0.shape[1]))
                    k = random.randint(0, len(files) - 1)
                    img1 = plt.imread(f'{i}/{files[k]}')
                    img1 = cv2.resize(img1, dsize=size)
                    img1 = np.reshape(img1, (1, img1.shape[0], img1.shape[1]))
                    img0 = np.array(img0, dtype='uint8')
                    img1 = np.array(img1, dtype='uint8')
                    same.append(np.asarray([img0, img1]).astype('uint8'))
        return same

    def false(f):
        false = []
        for i in range(len(f)):
            files = []
            for aa in range(repeat_f):
                for root, dirs, files in os.walk(f[i]):
                    pass
                for t in range(len(files)):
                    img0 = plt.imread(f'{f[i]}/{files[t]}')
                    img0 = cv2.resize(img0, dsize=size)
                    img0 = np.reshape(img0, (1, img0.shape[0], img0.shape[1]))
                    img0 = np.array(img0, dtype='uint8')
                    k = random.randint(0, len(f) - 1)
                    while f[i] == f[k]:
                        k = random.randint(0, len(f) - 1)
                    files1 = []
                    for root1, dirs1, files1 in os.walk(f[i]):
                        pass
                    k1 = random.randint(0, len(files1) - 1)
                    img1 = plt.imread(f'{f[k]}/{files1[k1]}')
                    img1 = cv2.resize(img1, dsize=size)
                    img1 = np.reshape(img1, (1, img1.shape[0], img1.shape[1]))
                    img1 = np.array(img1, dtype='uint8')
                    false.append(np.asarray([img0, img1]).astype('uint8'))
        return false
    f = []
    for root, dirs, files in os.walk(dir):
        f.append(root)
    f = f[1:]
    print(f)
    # same
    same = same(f)
    same = np.asarray(same).astype('uint8')
    print(same.shape)
    false = false(f)
    false = np.asarray(false).astype('uint8')
    y_same = np.ones(same.shape[0]).astype('uint8')
    y_false = np.zeros(false.shape[0]).astype('uint8')
    print(false.shape)
    X = np.concatenate((same, false))
    Y = np.concatenate((y_same, y_false))
    from sklearn.utils import shuffle
    X, Y = shuffle(X, Y, random_state=0)
    # cheaker(X, Y)
    np.save('data/X', X)
    np.save('data/Y', Y)


prepare(dir)













