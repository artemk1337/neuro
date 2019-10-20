import numpy as np
from numpy import genfromtxt
import os
from sys import exit
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, TerminateOnNaN


batch_size = 128


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


final_arr = np.concatenate((false_arr[9 * 1748:], true_arr))

i = 0
for i in range(9):
    final_arr = np.concatenate((final_arr, false_arr[i * 1609:(i + 1) * 1609]))
    final_arr = np.concatenate((final_arr, true_arr))

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


epochs = 200
seq = Sequential()
seq.add(Dense(16, input_dim=8, activation='relu'))
seq.add(Dense(8, activation='relu'))
seq.add(Dense(1, activation='sigmoid'))


seq.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = seq.fit(tr_data[:], tr_label[:], batch_size=batch_size,
                  epochs=epochs, verbose=1,
                  callbacks=[ModelCheckpoint('weights.hdf5',
                                             monitor='val_acc',
                                             verbose=1,
                                             save_best_only=True),
                             TerminateOnNaN()],
                  validation_data=(va_data[:], va_label[:]),
                  shuffle=True)
seq.load_weights(f'weights.hdf5')
seq.save(f'model')
print(seq.evaluate(te_data, te_label))
print(seq.predict(te_data[0:10]))
print(seq.predict(te_data[80:90]))


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


