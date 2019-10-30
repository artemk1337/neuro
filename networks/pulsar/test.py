import numpy as np
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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor


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


batch_size = 128
metr = 'acc'  # f1, AUC


def load():
    global tr_data, tr_label, va_data, va_label, te_data, te_label
    data = genfromtxt('pulsar_stars.csv', delimiter=',')[1:]
    false_arr = []
    true_arr = []
    i = 0
    for i in range(len(data[:])):
        if data[i, 8] == 0:
            false_arr.append(data[i])
        elif data[i, 8] == 1:
            true_arr.append(data[i])
    false_arr = np.asarray(false_arr)
    true_arr = np.asarray(true_arr)
    te_data, te_label = true_arr[:60, 0:8], true_arr[:60, 8]
    true_arr = true_arr[60:]
    te_data = np.concatenate((te_data, false_arr[:100, 0:8]))
    te_label = np.concatenate((te_label, false_arr[:100, 8]))
    false_arr = false_arr[100:]
    print(true_arr.shape)
    print(false_arr.shape)
    final_arr = np.concatenate((false_arr, true_arr))
    np.random.shuffle(final_arr)
    print(final_arr.shape)
    a = int(len(final_arr[:]) * 0.75)
    tr_data, tr_label = final_arr[:a, 0:8], final_arr[:a, 8]
    va_data, va_label = final_arr[a:, 0:8], final_arr[a:, 8]


"""epochs = 50
seq = Sequential()
seq.add(Dense(8, input_shape=tr_data[0].shape, activation='relu'))
seq.add(Dropout(0.25))
seq.add(Dense(6, activation='relu'))
seq.add(Dropout(0.25))
seq.add(Dense(4, activation='relu'))
seq.add(Dropout(0.25))
seq.add(Dense(1, activation='sigmoid'))"""


epochs = 100


def model():
    seq = Sequential()
    seq.add(Dense(16, input_dim=8, activation='relu'))
    seq.add(Dense(8, activation='relu'))
    seq.add(Dense(1, activation='sigmoid'))
    return seq


def train(seq):
    global history
    seq.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metr])
    checkpoint = ModelCheckpoint('weights.hdf5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)
    history = seq.fit(tr_data[:], tr_label[:], batch_size=batch_size,
                      epochs=epochs, verbose=2,
                      callbacks=[checkpoint,
                                 TerminateOnNaN(),
                                 ReduceLROnPlateau(monitor='val_loss',
                                                   factor=0.5,
                                                   patience=10)],
                      validation_data=(va_data[:], va_label[:]),
                      shuffle=True)
    # seq.load_weights(f'weights.hdf5')
    seq.save(f'model')
    print(f'predict: {seq.evaluate(te_data, te_label)}')
    print(seq.predict(te_data[0:20]))
    print(seq.predict(te_data[80:100]))


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
    load()
    m = model()
    train(m)
    ploting()


def rfc():
    g_perfect = []
    g_res = 0
    k_fail = 0
    k = 0
    g_max = 0
    for i in range(1, 301):
        print(f'i: {i}, k: {k}, g_res: {g_res}, g_max: {g_max}')
        g_res = 0
        for k in range(1, 501):
            if k_fail >= 50:
                k_fail = 0
                break
            clf = RandomForestClassifier(n_estimators=i, max_depth=k, random_state=0)
            clf.fit(tr_data, tr_label)
            tmp = clf.score(va_data, va_label)
            if tmp > g_res:
                k_fail = 0
                g_res = clf.score(va_data, va_label)
                if g_res > g_max:
                    g_max = g_res
                    g_perfect.append(i)
                    g_perfect.append(k)
            else:
                k_fail += 1
    return g_perfect[0], g_perfect[1]


def main1():
    load()
    i, k = rfc()
    clf = RandomForestClassifier(n_estimators=i, max_depth=k, random_state=0)
    clf.fit(tr_data, tr_label)
    print(clf.score(va_data, va_label))
    print(clf.predict(te_data[0:20]))
    print(clf.predict(te_data[80:100]))


def main2():
    load()
    """<=====Need_to_fix=====>"""
    rf = RandomForestRegressor(n_estimators=10, max_depth=100, random_state=0)
    clf = BaggingRegressor(rf, n_estimators=60, max_samples=0.1, random_state=25)
    """<=====================>"""
    clf.fit(tr_data, tr_label)
    print(clf.score(va_data, va_label))
    print(clf.predict(te_data[0:20]))
    print(clf.predict(te_data[80:100]))


# main()
main1()
# main2()
