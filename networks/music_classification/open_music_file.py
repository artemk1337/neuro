import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io.wavfile
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau
import tensorflow as tf


country_dir = 'data/country_wav'
rock_dir = 'data/rock_wav'


def rm(direct):
    f = []
    for roots, dirs, f in os.walk(direct):
        pass
    for i in f:
        htz, arr = scipy.io.wavfile.read(f"{direct}/{i}")
        if arr.shape[0] < 720000:
            os.remove(f"{direct}/{i}")
            print(f'remove - {i}')


def load(direct, arr1, y, d):
    f = []
    for roots, dirs, f in os.walk(direct):
        pass
    for i in f:
        print(i)
        htz, arr = scipy.io.wavfile.read(f"{direct}/{i}")
        ampl = 0
        k = 0
        for i in range(8):
            tmp = np.abs(arr[i * 100000:(i + 1) * 100000]).mean()
            if tmp >= ampl * 2:
                k = i
                break
            ampl = tmp
        arr = arr[k * 100000:(k + 1) * 100000]
        arr1.append(np.asarray(arr).astype('int'))
        y.append(d)


class Neuro:
    def __init__(self):
        pass

    def ploting(self, history):
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

    def model(self, input_dim):
        seq = Sequential()
        seq.add(Dense(10000, input_dim=input_dim, activation='relu'))
        seq.add(Dense(100, input_dim=input_dim, activation='relu'))
        seq.add(Dense(1, activation='relu'))
        return seq

    def train(self, x, y):
        x, x1, y, y1 = train_test_split(x, y, test_size=0.2)
        seq = self.model(x.shape[1])
        seq.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        checkpoint = ModelCheckpoint('weights.hdf5',
                                     monitor=f'val_acc',
                                     verbose=1,
                                     save_best_only=True)
        history = seq.fit(x[:], y[:],
                          batch_size=64,
                          epochs=300,
                          verbose=1,
                          callbacks=[checkpoint,
                                     TerminateOnNaN(),
                                     ReduceLROnPlateau(monitor='val_loss',
                                                       factor=0.7,
                                                       patience=10)],
                          validation_data=(x1[:], y1[:]),
                          shuffle=True)
        seq.load_weights(f'weights.hdf5')
        seq.save(f'model.h5')
        self.ploting(history)


'''# rm(country_dir)
# rm(rock_dir)
arr1 = []
y = []
load(country_dir, arr1, y, 0)
load(rock_dir, arr1, y, 1)
arr1 = np.asarray(arr1)
y = np.asarray(y).astype('uint8')
np.save('x', arr1)
np.save('y', y)
quit()'''
x = np.load('x.npy')
print(x.shape)
y = np.load('y.npy')
Neuro().train(x, y)









