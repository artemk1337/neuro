import numpy as np
import json
import csv
import h5py
from numpy import genfromtxt
import os
from sys import exit
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.metrics import Recall, Precision, AUC
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau
from keras.optimizers import Adam

import neuro


"""_SKLEARN_"""


import create_best_tree

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor


def load_data_1():
    table = []
    with open('data.csv', "r", newline="") as file:
        t = csv.reader(file)
        for row in t:
            table.append(row)
    table = table[1:]
    for i in range(len(table)):
        del table[i][:4]
        for k in range(len(table[i])):
            try:
                table[i][k] = float(table[i][k])
            except:
                table[i][k] = float(0)
    tmp = np.zeros((len(table), len(table[0])))
    for i in range(len(table)):
        tmp[i] = np.asarray(table[i], dtype=np.float)
    np.random.shuffle(tmp)
    y = tmp[:, 0]
    for i in range(y.shape[0]):
        if y[i] >= 0.5:
            y[i] = 1
        else:
            y[i] = 0
    x = tmp[:, 1:]
    print(tmp.shape)
    b = int(tmp.shape[0] * 0.75)
    tr_x, tr_y = x[:b], y[:b]
    te_x, te_y = x[b:], y[b:]
    print(tr_x.shape, tr_y.shape, te_x.shape, te_y.shape)
    return tr_x, tr_y, te_x, te_y


def main():
    a, b, c, d = load_data_1()
    m = neuro.create_model()
    neuro.train(a, b, c, d, m)
    neuro.ploting()


def main1(s):
    a, b, c, d = load_data_1()
    """Test"""
    logreg_clf = LogisticRegression()
    logreg_clf.fit(a, b)
    print(logreg_clf.score(c, d))
    dtr = DecisionTreeClassifier()
    dtr.fit(a, b)
    print(dtr.score(c, d))
    clf = RandomForestClassifier()
    clf.fit(a, b)
    print(clf.score(c, d))
    quit()
    """End_Test"""
    create_best_tree.rfc(a, b, c, d, s)


"""Neuro"""
# main()
"""Random forest"""
main1('rf.pkl')
print('\n\n\n')
main1('rf1.pkl')
print('\n\n\n')
main1('rf2.pkl')
print('\n\n\n')
main1('rf3.pkl')
