import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPool2D, Input, Lambda,\
    InputLayer, Conv1D, MaxPool1D, BatchNormalization, DepthwiseConv2D, SeparableConv2D
from keras.models import Model, Sequential
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau
# from keras.applications import MobileNetV2
# import cv2


size = 100
applications_model = 0


def data():
    x = np.load('data/x_tr_wb.npy').astype('float16')/255
    print(x.shape)
    y = np.load('data/y_tr_wb.npy')
    x_te = np.load('data/x_te_wb.npy').astype('float16')/255
    y_te = np.load('data/y_te_wb.npy')
    return x, y, x_te, y_te


def create_network(input_shape, i):
    seq = Sequential()

    x = 30
    nb = [x, 2*x, 3*x, 4*x, 5*x]

    # conv layer 1
    seq.add(Conv2D(nb[0], (3, 3), input_shape=input_shape, activation='relu'))
    # seq.add(BatchNormalization())
    seq.add(MaxPool2D())
    # seq.add(Dropout(.2))

    # conv layer 2
    seq.add(Conv2D(nb[1], (3, 3), activation='relu'))
    # seq.add(BatchNormalization())
    seq.add(MaxPool2D())
    # seq.add(Dropout(.2))

    # conv layer 3
    seq.add(Conv2D(nb[2], (3, 3), activation='relu'))
    # seq.add(BatchNormalization())
    seq.add(MaxPool2D())
    # seq.add(Dropout(.2))

    # conv layer 4
    seq.add(Conv2D(nb[3], (3, 3), activation='relu'))
    # seq.add(BatchNormalization())
    seq.add(MaxPool2D())
    # seq.add(Dropout(.2))

    # flatten
    seq.add(Flatten())
    seq.add(Dense(512, activation='relu'))
    # seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    seq.summary()
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


def compute_false_false(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    tmp = 0
    tmp1 = 0
    for i in range(pred.shape[0]):
        if y_true[i] == 0:
            tmp += 1
        if pred[i] == 0:
            tmp1 += 1
    if tmp > tmp1:
        return tmp1/tmp
    else:
        return 1


def f(y_true, y_pred):
    pred = K.cast(y_pred < 0.5, y_true.dtype)
    tmp = 0
    tmp1 = 0
    for i in range(pred.shape[0]):
        if y_true[i] == 0:
            tmp += 1
        if pred[i] == 0:
            tmp1 += 1
    if tmp > tmp1:
        k = tmp1/tmp
    else:
        k = 1
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype))) * 0.3 + 0.7 * k




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


def train(x, y, x1, y1):
    global history
    input_shape = (size, size, 1)
    batch_size = 256
    epochs = 50
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
                                                     factor=0.25,
                                                     patience=2)],
                        validation_data=([x1[:, 0], x1[:, 1]], y1))
    model.save(f'model_faces_wb')
    y_pred = model.predict([x[:, 0], x[:, 1]])
    tr_acc = compute_accuracy(y, y_pred)
    y_pred = model.predict([x1[:, 0], x1[:, 1]])
    te_acc = compute_accuracy(y1, y_pred)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    ploting()
    return model


def main():
    x, y, x1, y1 = data()
    model = train(x, y, x1, y1)


main()
