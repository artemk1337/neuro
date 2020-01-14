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


def create_model():
    seq = Sequential()
    seq.add(Dense(16, input_dim=874, activation='relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(10, activation='relu'))
    seq.add(Dense(1, activation='sigmoid'))
    return seq


metr = 'acc'
loss = 'binary_crossentropy'
optimizer = 'adam'
batch_size = 1028
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
                                                   patience=10)],
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

