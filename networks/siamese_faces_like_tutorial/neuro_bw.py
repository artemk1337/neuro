import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import keras
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPool2D, Input, Lambda,\
    InputLayer, Conv1D, MaxPool1D, MaxPooling2D, Convolution2D, Activation
from keras.models import Model, Sequential
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau
from keras.applications import MobileNetV2


size = (1, 56, 46)
applications_model = 0


def data():
    x = np.load('data/X.npy').astype('float16')/255
    print(x.shape)
    y = np.load('data/Y.npy')
    return x, y


def create_network(input_shape, i):
    seq = Sequential()

    nb_filter = [6, 12]
    kernel_size = 3

    # convolutional layer 1
    seq.add(Convolution2D(nb_filter[0], kernel_size, kernel_size, input_shape=input_shape,
                          border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    seq.add(Dropout(.25))

    # convolutional layer 2
    seq.add(Convolution2D(nb_filter[1], kernel_size, kernel_size, border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
    seq.add(Dropout(.25))

    # flatten
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq


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


def train(x, y):
    global history
    input_shape = size
    batch_size = 256
    epochs = 13
    network = create_network(input_shape, applications_model)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    # Photo to vector
    vect_a = network(input_a)
    vect_b = network(input_b)
    # Func for distance
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([vect_a, vect_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)
    checkpointer = ModelCheckpoint(filepath=f"data/best_weights.hdf5",
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True)
    model.compile(loss=contrastive_loss, optimizer='adam', metrics=[accuracy])
    history = model.fit([x[:, 0], x[:, 1]], y,
                        shuffle=True,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[checkpointer, TerminateOnNaN(),
                                   ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.6,
                                                     patience=2)],
                        validation_split=.25)
    model.save(f'data/model_faces_wb')
    '''y_pred = model.predict([x[:, 0], x[:, 1]])
    tr_acc = compute_accuracy(y, y_pred)
    y_pred = model.predict([x1[:, 0], x1[:, 1]])
    te_acc = compute_accuracy(y1, y_pred)'''

    """print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))"""
    ploting()
    return model


def main():
    x, y = data()
    model = train(x, y,)


main()
