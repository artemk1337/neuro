import numpy as np
import json
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


def load_data():
    with open('match_train.json', 'r', encoding='utf-8') as file:
        a = json.load(file)
    table = []
    # print(a['0'])
    for i in a:
        tmp = []
        """if a[i]['res'] >= 0.5:
            tmp.append(1)
        else:
            tmp.append(0)"""
        tmp.append(a[i]['res'])
        tmp.append(a[i]['rank_1'])
        tmp.append(a[i]['rank_2'])
        for k in a[i]['score_1']:
            for t in k:
                tmp.append(k[t])
        for k in a[i]['ranks_1']:
            for t in k:
                tmp.append(k[t])
        for k in a[i]['score_2']:
            for t in k:
                tmp.append(k[t])
        for k in a[i]['ranks_2']:
            for t in k:
                tmp.append(k[t])
        for k in a[i]['player_stat_1']:
            for t in k:
                tmp.append(k[t])
        for k in a[i]['player_stat_2']:
            for t in k:
                tmp.append(k[t])
        table.append(tmp)
        del tmp
    table = np.asarray(table).astype(np.float)
    np.random.shuffle(table)
    a = int(table.shape[0] * 0.8)
    tr_y = table[:a, 0]
    tr_x = table[:a, 1:]
    te_y = table[a:, 0]
    te_x = table[a:, 1:]
    del table
    print(tr_x.shape)
    return tr_x, tr_y, te_x, te_y


def create_model():
    seq = Sequential()
    seq.add(Dense(1024, input_dim=32, activation='relu'))
    seq.add(Dense(512, activation='relu'))
    seq.add(Dense(1, activation='sigmoid'))
    return seq


metr = 'mae'
# loss = 'binary_crossentropy'
loss = 'mse'
optimizer = 'adam'
batch_size = 10000
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
    a, b, c, d = load_data()
    m = create_model()
    train(a, b, c, d, m)
    ploting()


def rfc1(a, b, c, d):
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
        if i_fail >= 60:
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
                with open('rf.pkl', 'wb') as f:
                    cPickle.dump(clf, f)
            else:
                k_fail += 1
        i_fail += 1
    return g_perfect[0], g_perfect[1]


def main1():
    a, b, c, d = load_data()
    i, k = rfc1(a, b, c, d)
    clf = RandomForestClassifier(n_estimators=i, max_depth=k, random_state=0)
    clf.fit(a, b)
    print(clf.score(c, d))


"""Neuro"""
main()
"""Random forest"""
# main1()
