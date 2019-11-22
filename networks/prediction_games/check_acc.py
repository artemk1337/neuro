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
from keras import backend as K
from sklearn.metrics import roc_auc_score
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor
import _pickle as cPickle


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def roc_auc_score(y_true, y_pred):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("RocAucScore"):
        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))
        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)
        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p = 3
        difference = tf.zeros_like(pos * neg) + pos - neg - gamma
        masked = tf.boolean_mask(difference, difference < 0.0)
    return tf.reduce_sum(tf.pow(-masked, p))


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


a, b, c, d = load_data_1()

clf = cPickle.load(open('data/models/rf.pkl', 'rb'))

m = keras.models.load_model('data/models/model')

print(f'neuro - {int(m.evaluate(a[:], b[:])[1] * 100)}%')
print(f'neuro - {int(m.evaluate(c[:], d[:])[1] * 100)}%')
print(f'forest tr - {int(clf.score(a, b) * 100)}%')
print(f'forest te - {int(clf.score(c, d) * 100)}%')


# quit()


clf_test = cPickle.load(open('data/models/test/rf.pkl', 'rb'))
print(f'forest tr new - {int(clf_test.score(a, b) * 100)}%')
print(f'forest te new - {int(clf_test.score(c, d) * 100)}%')
