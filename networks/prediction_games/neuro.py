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
from keras.optimizers import Adam

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


def create_model():
    seq = Sequential()
    seq.add(Dense(100, input_dim=874, activation='relu'))
    seq.add(Dense(50, activation='relu'))
    seq.add(Dense(1, activation='sigmoid'))
    return seq


metr = 'acc'
loss = 'binary_crossentropy'
optimizer = 'RMSprop'
batch_size = 64
epochs = 500


def train(tr_x, tr_y, te_x, te_y, seq):
    global history
    seq.compile(loss=loss, optimizer=optimizer, metrics=[metr])
    checkpoint = ModelCheckpoint('weights.hdf5',
                                 monitor=f'val_{metr}',
                                 verbose=1,
                                 save_best_only=True)
    history = seq.fit(tr_x[:], tr_y[:], batch_size=batch_size,
                      epochs=epochs, verbose=2,
                      callbacks=[checkpoint,
                                 TerminateOnNaN(),
                                 ReduceLROnPlateau(monitor='val_loss',
                                                   factor=0.5,
                                                   patience=100)],
                      validation_data=(te_x[:], te_y[:]),
                      shuffle=True)
    # seq.load_weights(f'weights.hdf5')
    seq.save(f'model')


def ploting():
    # print(history.history.keys())
    ac = []
    for i in history.history.keys():
        ac.append(i)
    loss = history.history[ac[2]]
    val_loss = history.history[ac[0]]
    acc = history.history[ac[3]]
    val_acc = history.history[ac[1]]
    epochs = range(1, len(loss) + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(epochs, loss, 'bo', label='Training loss')
    ax1.plot(epochs, val_loss, 'b', label='Validation loss', color='r')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(epochs, acc, 'bo', label='Training acc')
    ax2.plot(epochs, val_acc, 'b', label='Validation acc', color='r')
    ax2.set_title('Training and validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    for ax in fig.axes:
        ax.grid(True)
    plt.savefig('graph')
    plt.show()


def main():
    a, b, c, d = load_data_1()
    m = create_model()
    train(a, b, c, d, m)
    ploting()


def rfc1(a, b, c, d, s):
    g_perfect = [1, 1]
    k_fail = 0
    i_fail = 0
    k = 0
    g_max = 0
    g_max_te = 0
    for i in range(1, 301):
        with open('max_res.json', 'w') as file:
            json.dump({"i": g_perfect[0], "k": g_perfect[1], "g_max": g_max, "g_max_te": g_max_te}, file)
        print(f'i: {i}, k: {k},\n'
              f'g_max: {g_max}, g_max_te: {g_max_te}')
        if i_fail >= 30:
            break
        prev_good_te = 0
        for k in range(1, 501):
            if k_fail >= 30:
                k_fail = 0
                # i_fail += 1
                # k += 30
                break
            clf = RandomForestClassifier(n_estimators=i, max_depth=k, random_state=0)
            clf.fit(a, b)
            tmp = clf.score(a, b)
            tmp1 = clf.score(c, d)
            if tmp1 > prev_good_te:
                k_fail = 0
                prev_good_te = tmp1
            if g_max_te < tmp1 <= tmp:
                g_max = tmp
                g_max_te = tmp1
                i_fail = 0
                k_fail = 0
                g_perfect[0] = i
                g_perfect[1] = k
                with open(s, 'wb') as f:
                    cPickle.dump(clf, f)
            else:
                k_fail += 1
        i_fail += 1
    return g_perfect[0], g_perfect[1]


def main1(s):
    a, b, c, d = load_data_1()
    i, k = rfc1(a, b, c, d, s)


"""Neuro"""
main()
"""Random forest"""
main1('rf.pkl')
main1('rf1.pkl')
main1('rf2.pkl')
main1('rf3.pkl')
