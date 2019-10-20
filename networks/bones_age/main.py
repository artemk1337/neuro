import os
from sys import exit
import shutil
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import time as tm
import keras
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPool2D, Input, Lambda, InputLayer, concatenate
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.utils import to_categorical
from keras import backend as K
from PIL import Image
import random
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau
import csv


size = 200


def dataset():
    if not os.path.exists(f'data/samples_{size}.npy'):
        inf = []
        with open('data/boneage-training-dataset.csv', newline='') as File:
            reader = csv.reader(File)
            for row in reader:
                inf.append(row)
        inf = np.asarray(inf)
        inf = inf[1:]
        i = 0
        while i < len(inf):
            if inf[i, 2] == 'False':
                inf[i, 2] = 1
            elif inf[i, 2] == 'True':
                inf[i, 2] = 0
            i += 1
        inf = inf.astype('uint16')
        print(inf)
        np.save('data/inf', inf)
        samples = []
        for i in inf[:, 0]:
            print(i)
            img = Image.open(f'data/boneage-training-dataset/{i}.png')\
                .resize((size, int(size * 1.35)),
                        Image.ANTIALIAS)
            arr = np.array(img, dtype='uint8').reshape((int(size * 1.35), size, 1))
            arr = arr.astype(dtype='float16') / 255
            """result = Image.fromarray((arr * 255).astype('uint8'))
            result.save('out_open1.jpg')"""
            samples.append(arr)
        samples = np.asarray(samples)
        np.save(f'data/samples_{size}', samples)
    else:
        samples = np.load(f'data/samples_{size}.npy')
        inf = np.load('data/inf.npy')
    return samples, inf


samples, inf = dataset()

a = int(len(samples) * 0.75)
tr, trl = samples[:a], inf[:a, 1:3]
te, tel = samples[a:], inf[a:, 1:3]
del samples, inf
print(tr.shape, te.shape, trl.shape, tel.shape)


epochs = 10
batch_size = 32

print(tr.shape)
input = Input(shape=(int(size * 1.35), size, 1), name='bones')
gender = Input(shape=(1,))
x = Conv2D(32, (3, 3), activation='relu')(input)
x = MaxPool2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPool2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPool2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPool2D((2, 2))(x)
x = Conv2D(512, (3, 3), activation='relu')(x)
x = MaxPool2D((2, 2))(x)
x = Conv2D(512, (3, 3), activation='relu')(x)
x = MaxPool2D((2, 2))(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(1, activation='relu')(x)
x = concatenate([x, gender])
x = Dense(8, activation='relu')(x)
age = Dense(1, activation='relu', name='age')(x)

model = Model([input, gender], age)
print(model.summary())


# checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_acc', verbose=1,save_best_only=True)
model.compile(optimizer='adam', loss=['mse'], metrics=['acc'])
history = model.fit([tr[:], trl[:, 1]], trl[:, 0], batch_size=batch_size,
                    epochs=epochs, verbose=1,
                    callbacks=[TerminateOnNaN(),
                               ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.4,
                                                 patience=50)],
                    validation_data=([te[:], tel[:, 1]], tel[:, 0]),
                    shuffle=True)

# model.load_weights(f'weights.hdf5')
model.save(f'model')
print(model.evaluate([te[:], tel[:, 1]], tel[:, 0]))


def ploting():
    print(history.history.keys())
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

