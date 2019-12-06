import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random


size = 100


def cheaker(res, target):
    for i in range(res.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(res[i, 0])
        ax1 = fig.add_subplot(122)
        ax1.imshow(res[i, 1])
        print(target[i])
        plt.show()


def create(dir, fn1, fn2, n1):
    def load_img(s1, s2):
        img0 = cv2.imread(f'{dir}/{s1}', cv2.IMREAD_GRAYSCALE)
        img0 = np.array(img0, dtype='uint8').reshape((size, size, 1))
        img1 = cv2.imread(f'{dir}/{s2}', cv2.IMREAD_GRAYSCALE)
        img1 = np.array(img1, dtype='uint8').reshape((size, size, 1))
        arr = np.asarray([img0, img1])
        return arr

    def same(files):
        print("<=====SAME=====>")
        if not os.path.exists(f'data/{fn1}_wb.npy'):
            res = []
            for i in range(len(files) - 1):
                try:
                    # print(files[i])
                    name = files[i].split('_')[1]
                    name1 = files[i + 1].split('_')[1]
                    if name == name1:
                        res.append(np.asarray(load_img(files[i], files[i + 1])))
                    k = random.randint(0, len(files) - 1)
                    name1 = files[k].split('_')[1]
                    while name != name1:
                        k = random.randint(0, len(files) - 1)
                        name1 = files[k].split('_')[1]
                    res.append(np.asarray(load_img(files[i], files[k])))
                except:
                    pass
            res = np.asarray(res).astype('uint8')
            np.random.shuffle(res)
            np.save(f'data/{fn1}_wb', res)
            # cheaker(res)
            del res

    def diff(files):
        print("<=====DIFF=====>")
        if not os.path.exists(f'data/{fn2}_wb.npy'):
            res = []
            for i in range(len(files)):
                # print(files[i])
                try:
                    for _ in range(2):
                        name = files[i].split('_')[1]
                        k = random.randint(0, len(files) - 1)
                        name1 = files[k].split('_')[1]
                        while name == name1:
                            k = random.randint(0, len(files) - 1)
                            name1 = files[k].split('_')[1]
                        res.append(np.asarray(load_img(files[i], files[k])))
                except:
                    pass
            res = np.asarray(res).astype('uint8')
            np.random.shuffle(res)
            np.save(f'data/{fn2}_wb', res)
            # cheaker(res)
            del res

    f = []
    for roots, dirs, f in os.walk(dir):
        pass
    same(f)
    diff(f)
    a = np.load(f'data/{fn1}_wb.npy')
    l1 = a.shape[0]
    print(a.shape)
    b = np.load(f'data/{fn2}_wb.npy')
    print(b.shape)
    res = np.concatenate((a, b), axis=0)
    del a, b
    y = np.zeros(res.shape[0], dtype='uint8')
    for i in range(l1):
        y[i] = 1
    from sklearn.utils import shuffle
    res, y = shuffle(res, y, random_state=0)
    np.save(f'data/y_{n1}_wb', y)
    del y
    np.save(f'data/x_{n1}_wb', res)
    del res
    # cheaker(np.load(f'data/x_{n1}_wb.npy'), np.load(f'data/y_{n1}_wb.npy'))


def prepare_for_arr():
    def prepare(dir, dsave, dsave1):
        files = []
        for roots, dirs, files in os.walk(dir):
            pass
        for i in range(int(len(files) * 0.8)):
            print(i)
            try:
                if not os.path.exists(f'{dsave}/{files[i]}'):
                    img = cv2.imread(f'{dir}/{files[i]}', cv2.IMREAD_GRAYSCALE)
                    plt.imsave(f'{dsave}/{files[i]}',
                               cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_AREA),
                               cmap='gray')
            except:
                print(f'ERROR - {files[i]}')
        for i in range(int(len(files) * 0.8), len(files)):
            print(i)
            try:
                if not os.path.exists(f'{dsave1}/{files[i]}'):
                    img = cv2.imread(f'{dir}/{files[i]}', cv2.IMREAD_GRAYSCALE)
                    plt.imsave(f'{dsave1}/{files[i]}',
                               cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_AREA),
                               cmap='gray')
            except:
                print(f'ERROR - {files[i]}')

    if not os.path.exists('data/final_wb'):
        os.makedirs('data/final_wb')
        os.makedirs('data/final_test_wb')
    prepare('data/male_cut', 'data/final_wb', 'data/final_test_wb')
    if not os.path.exists('data/final_wb'):
        os.makedirs('data/final_wb')
        os.makedirs('data/final_test_wb')
    prepare('data/female_cut', 'data/final_wb', 'data/final_test_wb')


def cut_before_prepare():
    def cut(dir_name, dir_name_cut, n):
        files = []
        for roots, dirs, files in os.walk(dir_name):
            pass

        data = np.loadtxt(n, delimiter='\t', dtype=str)
        data = data[1:]

        for i in files:
            [counter, name] = i.split('_')
            x1, y1, x2, y2 = data[int(counter), 4].split(',')
            try:
                if not os.path.exists(f'{dir_name_cut}/{i}'):
                    img = plt.imread(rf'{dir_name}/{i}')
                    img = img[int(y1):int(y2), int(x1):int(x2)]
                    plt.imsave(f'{dir_name_cut}/{i}', img)
                    print(f'SUCCESS - {counter}')
            except:
                pass

        for i in files:
            try:
                if os.path.getsize(f'{dir_name_cut}/{i}') <= 2500:
                    os.remove(f'{dir_name_cut}/{i}')
                    print(f'remove - {i}')
            except:
                pass

    if not os.path.exists('data/male_cut'):
        os.makedirs('data/male_cut')
    cut('data/male', 'data/male_cut', 'facescrub_actors.txt')
    if not os.path.exists('data/female_cut'):
        os.makedirs('data/female_cut')
    cut('data/female', 'data/female_cut', 'facescrub_actresses.txt')


# cut_before_prepare()
prepare_for_arr()
create('data/final_wb', 'same', 'diff', 'tr')
create('data/final_test_wb', 'same_te', 'diff_te', 'te')



