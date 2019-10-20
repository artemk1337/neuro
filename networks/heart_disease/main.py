import numpy as np
from numpy import genfromtxt
import os
from sys import exit
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau


data = genfromtxt('heart.csv', delimiter=',')[1:]

print(data.shape)

np.random.shuffle(data)

a = int(len(data) * 0.75)

tr = data[:a]
te = data[a:]

del data

np.random.shuffle(tr)
np.random.shuffle(te)

trl = tr[:, 13]
tr = tr[:, 0:13]

tel = te[:, 13]
te = te[:, 0:13]


epochs = 1000
batch_size = 1024

seq = Sequential()
seq.add(Dense(13, input_dim=13, activation='relu'))
seq.add(Dense(1, activation='sigmoid'))


seq.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = seq.fit(tr[:], trl[:], batch_size=batch_size,
                  epochs=epochs, verbose=1,
                  callbacks=[ModelCheckpoint('weights.hdf5',
                                             monitor='val_acc',
                                             verbose=1,
                                             save_best_only=True),
                             TerminateOnNaN(),
                             ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.4,
                                               patience=50)],
                  validation_data=(te[:], tel[:]),
                  shuffle=True)

seq.load_weights(f'weights.hdf5')
seq.save(f'model')
print(seq.evaluate(te, tel))


def ploting():
    # print(history.history.keys())
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
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


ploting()

