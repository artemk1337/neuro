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
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPool2D, Input, Lambda, InputLayer, Conv1D, MaxPool1D
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.utils import to_categorical
from keras import backend as K
from PIL import Image
import random
from keras.callbacks import ModelCheckpoint, TerminateOnNaN
from keras.applications.mobilenet_v2 import MobileNetV2
# from livelossplot.keras import PlotLossesCallback


RGB = 1  # Black and white mode: 1=On or 3=Off.
size = 120
epochs = 15
batch_size = 512


if RGB != 3 and RGB != 1:
    exit(-1)


time1 = tm.time()


'''<===============PREPARATION_DATA===============>'''


def img_open(d, root, i, k):
    input_shape = (size, size, RGB)
    if RGB == 3:
        img = Image.open(f'{root[i + 1]}/{d[root[i + 1]][k]}') \
                            .resize((size, size), Image.ANTIALIAS)
    else:
        img = Image.open(f'{root[i + 1]}/{d[root[i + 1]][k]}').convert('L') \
            .resize((size, size), Image.ANTIALIAS)
    arr = np.array(img, dtype='uint8').reshape(input_shape)
    """result = Image.fromarray((arr).astype('uint8'))
    result.save('out_open.jpg')"""
    arr = arr.astype(dtype='float16') / 255
    """result = Image.fromarray((arr * 255).astype('uint8'))
    result.save('out_open1.jpg')"""
    return arr


def preparation_data_true():
    print('RIGHT PARE')
    d, root = preparation_data()
    truearr = []
    i = 0
    for i in range(100):
        k = 0
        print(f'i: {i}')
        while k < (len(d[root[i + 1]]) - 1):
            truearr1 = [img_open(d, root, i, k),
                        img_open(d, root, i, k + 1)]
            truearr.append(truearr1)
            del truearr1
            if size < 200 and RGB == 1:
                truearr1 = [img_open(d, root, i, k),
                            img_open(d, root, i, random.randint(0, len(d[root[i + 1]]) - 1))]
                truearr.append(truearr1)
                del truearr1
            if size < 128 and RGB == 1:
                truearr1 = [img_open(d, root, i, k),
                            img_open(d, root, i, random.randint(0, len(d[root[i + 1]]) - 1))]
                truearr.append(truearr1)
                del truearr1
            if size < 128 and RGB == 1:
                truearr1 = [img_open(d, root, i, k),
                            img_open(d, root, i, random.randint(0, len(d[root[i + 1]]) - 1))]
                truearr.append(truearr1)
                del truearr1
            k += 1
    truearr = np.array(truearr)
    print(truearr.shape)
    if not os.path.exists('data'):
        os.makedirs('data')
    np.save(f'data/truearr_{RGB}_{size}', truearr)
    del d, root
    del truearr


def preparation_data_false():
    print('WRONG PARE')
    d, root = preparation_data()
    wrongarr = []
    i = 0
    for i in range(100):
        k = 0
        print(f'WRONG i: {i}')
        while k < (len(d[root[i + 1]]) - 1):
            t = random.randint(1, 100)
            while t == i:
                t = random.randint(1, 100)
            wrongarr1 = [img_open(d, root, i, k),
                         img_open(d, root, t - 1, random.randint(0, len(d[root[t]]) - 1))]
            wrongarr.append(wrongarr1)
            del wrongarr1
            if size < 200 and RGB == 1:
                t = random.randint(1, 100)
                while t == i:
                    t = random.randint(1, 100)
                wrongarr1 = [img_open(d, root, i, k),
                             img_open(d, root, t - 1, random.randint(0, len(d[root[t]]) - 1))]
                wrongarr.append(wrongarr1)
                del wrongarr1
            if size < 128 and RGB == 1:
                t = random.randint(1, 100)
                while t == i:
                    t = random.randint(1, 100)
                wrongarr1 = [img_open(d, root, i, k),
                             img_open(d, root, t - 1, random.randint(0, len(d[root[t]]) - 1))]
                wrongarr.append(wrongarr1)
                del wrongarr1
                t = random.randint(1, 100)
                while t == i:
                    t = random.randint(1, 100)
                wrongarr1 = [img_open(d, root, i, k),
                             img_open(d, root, t - 1, random.randint(0, len(d[root[t]]) - 1))]
                wrongarr.append(wrongarr1)
                del wrongarr1
            k += 1
    wrongarr = np.array(wrongarr)
    print(wrongarr.shape)
    if not os.path.exists('data'):
        os.makedirs('data')
    np.save(f'data/falsearr_{RGB}_{size}', wrongarr)
    del d, root
    del wrongarr


def preparation_data():
    dir_name = 'data/PINS'
    root = []  # Директории
    d = {}  # Словарь с директориями и файлами
    for roots, dirs, files in os.walk(dir_name):
        root.append(roots)
        for t in files:
            d.update({roots: files})
    if not os.path.exists('data'):
        os.makedirs('data')
    with open("data/faces.json", "w", encoding="utf-8") as file:
        json.dump(d, file, indent=4, separators=(',', ': '))
    return d, root


def data():
    if not (os.path.exists(f'data/truearr_SHUFFLE_{RGB}_{size}.npy') or
            os.path.exists(f'data/labels_SHUFFLE_{RGB}_{size}.npy') or
            os.path.exists(f'data/test_a_SHUFFLE_{RGB}_{size}.npy') or
            os.path.exists(f'data/test_labels_SHUFFLE_{RGB}_{size}.npy')):
        if not os.path.exists(f'data/truearr_{RGB}_{size}.npy'):
            preparation_data_true()
        if not os.path.exists(f'data/falsearr_{RGB}_{size}.npy'):
            preparation_data_false()


'''<===============FUNCTIONS_FROM_KERAS_MANUAL===============>'''


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


""""<===============NETWORK===============>"""


def ploting():
    # print(history.history.keys())
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
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


def create_network(input_shape):
    seq = Sequential()
    seq.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    seq.add(MaxPool2D((2, 2)))
    seq.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    seq.add(MaxPool2D((2, 2)))
    seq.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    seq.add(MaxPool2D((2, 2)))
    seq.add(Flatten())
    seq.add(Dense(1024, activation='relu'))
    seq.add(Dense(128, activation='relu'))
    print(seq.summary())
    return seq


def test(train_d, train_l):
    while True:
        a = int(input('input: '))
        a1 = train_d[a, 0]
        result = Image.fromarray((a1 * 255).astype('uint8'))
        result.save('out_0.jpg')
        a1 = train_d[a, 1]
        result = Image.fromarray((a1 * 255).astype('uint8'))
        result.save('out_1.jpg')
        print(train_l[a])


def train(train_d, train_l, test_d, test_l):
    global history
    # test(train_d, train_l)
    # Constants
    if RGB == 3:
        input_shape = (size, size, RGB)
    else:
        input_shape = (size, size, RGB)
    # Create network
    network = create_network(input_shape)
    # Input shape for photo
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    # Photo to vector
    vect_a = network(input_a)
    vect_b = network(input_b)
    # Func for distance
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([vect_a, vect_b])
    # Create Model
    model = Model(inputs=[input_a, input_b], outputs=distance)
    checkpointer = ModelCheckpoint(filepath=f"best_weights_for_faces_{RGB}.hdf5",
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True)
    # model.compile(loss=contrastive_loss, optimizer=RMSprop(), metrics=[accuracy])
    model.compile(loss=contrastive_loss, optimizer=RMSprop(), metrics=[accuracy])
    history = model.fit([train_d[:, 0], train_d[:, 1]], train_l,
                        shuffle=True,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[checkpointer, TerminateOnNaN()],
                        validation_data=([test_d[:, 0], test_d[:, 1]], test_l))
    # Final prediction
    # model.load_weights(f'best_weights_for_faces_{RGB}.hdf5')
    model.save(f'data/model_faces_{RGB}_{size}')
    y_pred = model.predict([train_d[:, 0], train_d[:, 1]])
    tr_acc = compute_accuracy(train_l, y_pred)
    y_pred = model.predict([test_d[:, 0], test_d[:, 1]])
    te_acc = compute_accuracy(test_l, y_pred)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    del te_acc, test_d, test_l, train_d, train_l, tr_acc
    ploting()


def shuffle(a, labels, tr_max):
    i = 0
    print(f'Shape arr: {a.shape}')
    print(f'Labels arr: {labels.shape}')
    while i < tr_max:
        k = random.randint(0, tr_max - 1)
        while k == i:
            k = random.randint(0, tr_max - 1)
        a[i], a[k] = a[k], a[i].copy()
        del k
        k = random.randint(tr_max, (tr_max * 2) - 1)
        while k == (i + tr_max):
            k = random.randint(tr_max, (tr_max * 2) - 1)
        a[i + tr_max], a[k] = a[k], a[i + tr_max].copy()
        del k
        i += 1
    i = 0
    while i < (tr_max * 2):
        k = random.randint(0, (tr_max * 2) - 1)
        while k == i:
            k = random.randint(0, (tr_max * 2) - 1)
        a[i], a[k] = a[k], a[i].copy()
        labels[i], labels[k] = labels[k], labels[i].copy()
        i += 1


def load_data():
    if os.path.exists(f'data/truearr_SHUFFLE_{RGB}_{size}.npy') and \
            os.path.exists(f'data/labels_SHUFFLE_{RGB}_{size}.npy') and \
            os.path.exists(f'data/test_a_SHUFFLE_{RGB}_{size}.npy') and \
            os.path.exists(f'data/test_labels_SHUFFLE_{RGB}_{size}.npy'):
        a = np.load(f'data/truearr_SHUFFLE_{RGB}_{size}.npy')
        print(f'Shape arr: {a.shape}')
        labels = np.load(f'data/labels_SHUFFLE_{RGB}_{size}.npy')
        print(f'Shape labels: {labels.shape}')
        test_a = np.load(f'data/test_a_SHUFFLE_{RGB}_{size}.npy')
        print(f'Shape test: {test_a.shape}')
        test_labels = np.load(f'data/test_labels_SHUFFLE_{RGB}_{size}.npy')
        print(f'Shape test_labels: {test_labels.shape}')
        return a, labels, test_a, test_labels
    print('<=====OPEN_RIGHT_PARE=====>')
    a = np.load(f'data/truearr_{RGB}_{size}.npy')
    print(f'Shape arr: {a.shape}')
    max = len(a)
    tr_max = int(0.75 * len(a))
    print('<=====OPEN_WRONG_PARE=====>')
    b = np.load(f'data/falsearr_{RGB}_{size}.npy')
    print(f'Shape arr: {b.shape}')
    labels = np.zeros(max * 2)
    i = 0
    while i < max:
        labels[i] = 1
        i += 1
    a = np.concatenate((a, b))
    del b
    print('<=====SHUFFLE_DATA=====>')
    shuffle(a, labels, max)
    test_a = a[tr_max * 2:]
    a = a[:tr_max * 2]
    test_labels = labels[tr_max * 2:]
    labels = labels[:tr_max * 2]
    np.save(f'data/truearr_SHUFFLE_{RGB}_{size}.npy', a)
    np.save(f'data/labels_SHUFFLE_{RGB}_{size}.npy', labels)
    np.save(f'data/test_a_SHUFFLE_{RGB}_{size}.npy', test_a)
    np.save(f'data/test_labels_SHUFFLE_{RGB}_{size}.npy', test_labels)
    del i
    return a, labels, test_a, test_labels


def main():
    data()
    train_d, train_l, test_d, test_l = load_data()
    train(train_d, train_l, test_d, test_l)
    print(f'Время работы программы - {tm.time() - time1}')


main()


