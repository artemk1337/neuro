import tree
import warnings
import os
import sys
import shutil
import numpy as np
import random
import csv
import time
import json
from threading import Thread
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


g_trees = 50
max_depth = 10


def load_data(fn):
    x = []
    y = []
    with open(fn, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            top = np.asarray(row)
            break
        for row in csv_reader:
            x.append(np.asarray(row[0:5]))
            y.append(row[5])
    x, y = shuffle(np.asarray(x[1:]).astype('float32'),
                   np.asarray(y[1:]).astype('float32'),
                   random_state=0)
    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.2, random_state=0)
    del x, y
    new_data_x = []
    new_data_y = []
    for i in range(g_trees):
        tmp_x = []
        tmp_y = []
        for k in range(x1.shape[0]):
            t = random.randint(0, x1.shape[0] - 1)
            tmp_x.append(x1[t])
            tmp_y.append(y1[t])
        tmp_x = np.asarray(tmp_x)
        tmp_y = np.asarray(tmp_y)
        new_data_x.append(tmp_x)
        new_data_y.append(tmp_y)
    new_data_x = np.asarray(new_data_x)
    new_data_y = np.asarray(new_data_y)
    print(new_data_x.shape)
    return new_data_x, new_data_y, x1, x2, y1, y2, top


def load_data_predict(fn):
    x3 = []
    with open(fn, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            x3.append(np.asarray(row[0:5]))
    x3 = np.asarray(x3[1:]).astype('float32')
    return x3


def save(fn, y1, prediction_1, y2, prediction_2):
    text = {"train": np.mean(np.abs(y1 - prediction_1).astype('float64')),
            "test": np.mean(np.abs(y2 - prediction_2).astype('float64'))}
    print(text)
    with open(fn, 'w') as f:
        json.dump(text,
                  f, indent=4, separators=(',', ': '))


def plot(y_all, prediction_all):
    plt.plot(y_all - prediction_all)
    plt.axvline(x=int(0.8 * y_all.shape[0]), color='r')
    plt.legend(['data', 'split'])
    plt.savefig('redshift.png')
    # plt.show()


predictions = [[], [], []]


# А тут многопоточность не работает :c
# Хочу писать на плюсах
class Thread_tree(Thread):
    def __init__(self, name, new_data_x, new_data_y, i):
        Thread.__init__(self)
        self.name = name
        self.new_data_x = new_data_x
        self.new_data_y = new_data_y
        self.i = i

    def run(self):
        '''
        Use DecisionTreeRegressor because
        faster but work as good as tree.TR.
        '''
        rtf = tree.TR(max_depth=max_depth)
        # rtf = DecisionTreeRegressor(max_depth=max_depth)
        rtf.fit(new_data_x[self.i], new_data_y[self.i])
        predictions[0].append(np.asarray(rtf.predict(x1)))
        predictions[1].append(np.asarray(rtf.predict(x2)))
        predictions[2].append(np.asarray(rtf.predict(x3)))
        print(f'{self.name} - finish')


def predict(new_data_x, new_data_y):
    def create_threads():
        threads = []
        print('Just wait ~5 minutes :) ')
        # Create threads
        for i in range(g_trees):
            name = f"Tree №{i + 1}"
            my_thread = Thread_tree(name, new_data_x, new_data_y, i)
            my_thread.start()
            threads.append(my_thread)
        # Join threads
        for t in threads:
            t.join()

    create_threads()
    # time.sleep(5)
    prediction_1 = np.asarray(predictions[0]).mean(axis=0)
    prediction_2 = np.asarray(predictions[1]).mean(axis=0)
    prediction_3 = np.asarray(predictions[2]).mean(axis=0)
    return prediction_1, prediction_2, prediction_3


def save_csv(x3, prediction_3):
    res = []
    for i in range(x3.shape[0]):
        res.append(np.asarray([*x3[i], prediction_3[i]]))
    res = np.asarray(res)
    res = np.concatenate((np.asarray([top]), res))

    with open('sdss_predict.csv', "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for line in res:
            writer.writerow(line)


warnings.filterwarnings("ignore", category=RuntimeWarning)
tm = time.time()
new_data_x, new_data_y, x1, x2, y1, y2, top = load_data('sdss_redshift.csv')
x3 = load_data_predict('sdss.csv')
prediction_1, prediction_2, prediction_3 = predict(new_data_x, new_data_y)
plot(np.concatenate((y1, y2)), np.concatenate((prediction_1, prediction_2)))
save('redhsift.json', y1, prediction_1, y2, prediction_2)
save_csv(x3, prediction_3)
print(f'Время работы - {round(time.time() - tm, 2)} sec')
