import keras
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras import backend as K


size = 120


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
    pred = y_pred.ravel() < 0.5
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
    pred = y_pred.ravel() < 0.5
    tmp = 0
    tmp1 = 0
    for i in range(pred.shape[0]):
        if y_true[i] == 1:
            tmp += 1
        if pred[i] == 1:
            tmp1 += 1
    if tmp > tmp1:
        t = tmp1 / tmp
    else:
        t = 1
    return t * 0.5 + 0.5 * k


def cheaker(res):
    for i in range(res.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(res[i, 0])
        ax1 = fig.add_subplot(122)
        ax1.imshow(res[i, 1])
        plt.show()


def load():
    f = []
    for roots, dirs, f in os.walk('data/'):
        pass
    arr = []
    y = []

    def load_img(s1, s2):
        name1 = s1.split(',')[0]
        name2 = s2.split(',')[0]
        img0 = cv2.imread(f'data/test/{s1}', cv2.IMREAD_GRAYSCALE)
        img0 = cv2.resize(img0, dsize=(size, size), interpolation=cv2.INTER_AREA)
        img0 = np.array(img0, dtype='uint8')
        img1 = cv2.imread(f'data/test/{s2}', cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, dsize=(size, size), interpolation=cv2.INTER_AREA)
        img1 = np.array(img1, dtype='uint8')
        arr = np.asarray([img0, img1])
        if name1 == name2:
            y.append(1)
        else:
            y.append(0)
        return arr

    for i in f:
        for k in f:
            arr.append(np.asarray(load_img(i, k)))
    arr = np.asarray(arr).astype('uint8')/255
    return arr, np.asarray(y).astype('uint8')


model = keras.models.load_model('data/good/model_faces_wb', custom_objects={'contrastive_loss': contrastive_loss})


arr, y = load()

# print(model.predict([arr[1:2, 0], arr[1:2, 1]]))
print(y.shape)
print(arr.shape)

print(f'Samples - {arr.shape[0]}')

y_pred = model.predict([arr[:, 0], arr[:, 1]])
tr_acc = compute_false_false(y, y_pred)
print('* False -> False (like precision): %0.2f%%' % (100 * tr_acc))

y_pred = model.predict([arr[:, 0], arr[:, 1]])
tr_acc = compute_accuracy(y, y_pred)
print('* Avarage accuracy: %0.2f%%' % (100 * tr_acc))

y_pred = model.predict([arr[:, 0], arr[:, 1]])
tr_acc = f(y, y_pred)
print(tr_acc)
print('* F-metr: %0.2f%%' % (100 * tr_acc))


