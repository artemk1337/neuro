import numpy as np
import csv
import os
from sys import exit
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.metrics import Recall, Precision, AUC
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau
from keras import backend as K
from sklearn.metrics import roc_auc_score
import metrics

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor


"""

name            Object full name
a               semi-major axis(au)
e               eccentricity
i               Inclination with respect to x-y ecliptic plane(deg)
om              Longitude of the ascending node
w               argument of perihelion
q               perihelion distance(au)
ad              aphelion distance(au)
per_y           Orbital period(YEARS)
data_arc        data arc-span(d)
condition_code  Orbit condition code
n_obs_used      Number of observation used
H               Absolute Magnitude parameter
neo             Near Earth Object
pha             Physically Hazardous Asteroid
diameter        Diameter of asteroid(Km)
extent          Object bi/tri axial ellipsoid dimensions(Km)
albedo          geometric albedo
rot_per         Rotation Period(h)
GM              Standard gravitational parameter, Product of mass and gravitational constant
BV              Color index B-V magnitude difference
UB              Color index U-B magnitude difference
IR              Color index I-R magnitude difference
spec_B          Spectral taxonomic type(SMASSII)
spec_T          Spectral taxonomic type(Tholen)
G               Magnitude slope parameter
moid            Earth Minimum orbit Intersection Distance(au)
class           asteroid orbit class
n               Mean motion(deg/d)
per             orbital Period(d)
ma              Mean anomaly(deg)

much data:
2 - 16, 27 - 30 rows

"""


def load():
    if not os.path.exists('data/astro.npy') or not os.path.exists('data/astro_te.npy'):
        data = []
        with open('data/Asteroid.csv', 'r', newline='') as file:
            reader = csv.reader(file)
            for i in reader:
                data.append(i)
        data = np.asarray(data)
        data = data[1:]
        np.random.shuffle(data)
        i = int(data.shape[0] * 0.75)
        te_data = data[i:]
        data = data[:i]
        np.save('data/astro', data)
        np.save('data/astro_te', te_data)
        return data, te_data
    else:
        data = np.load('data/astro.npy')
        te_data = np.load('data/astro_te.npy')
        # 15 - diameter
        # print(te_data[16, 15])
        return data, te_data


def take_labels(tr, te):
    def show_max_mesh():
        score_tr = np.zeros(31, dtype=np.int)
        for i1 in range(tr.shape[0]):
            for k in range(31):
                if tr[i1, k]:
                    score_tr[k] += 1
        print(score_tr)  # Show max not empty mesh in each row
        score_te = np.zeros(31, dtype=np.int)
        for i1 in range(te.shape[0]):
            for k in range(31):
                if te[i1, k]:
                    score_te[k] += 1
        print(score_te)  # Show max not empty mesh in each row

    def prepare_1_15_rows():
        for i in range(tr.shape[0]):
            if tr[i, 13] == 'N':
                tr[i, 13] = 0
            elif tr[i, 13] == 'Y':
                tr[i, 13] = 1
            if tr[i, 14] == 'N':
                tr[i, 14] = 0
            elif tr[i, 14] == 'Y':
                tr[i, 14] = 1
        for i in range(te.shape[0]):
            if te[i, 13] == 'N':
                te[i, 13] = 0
            elif te[i, 13] == 'Y':
                te[i, 13] = 1
            if te[i, 14] == 'N':
                te[i, 14] = 0
            elif te[i, 14] == 'Y':
                te[i, 14] = 1
        for i in range(tr.shape[0]):
            for k in range(1, 16):
                try:
                    tr[i, k] = tr[i, k].astype(np.float)
                except Exception:
                    tr[i, k] = 0
        for i in range(te.shape[0]):
            for k in range(1, 16):
                try:
                    te[i, k] = te[i, k].astype(np.float)
                except Exception:
                    te[i, k] = 0

    # show_max_mesh()
    prepare_1_15_rows()
    tr_data, tr_label = tr[:, 1:15], tr[:, 15:16]
    del tr
    te_data, te_label = te[:, 1:15], te[:, 15:16]
    del te
    return tr_data, tr_label, te_data, te_label


def create_model(len_):
    m = Sequential()
    # 100, 64
    m.add(Dense(20, input_dim=len_, activation='relu'))
    m.add(Dense(10, activation='relu'))
    m.add(Dense(1, activation='relu'))
    return m


def training_neuro(tr_data, tr_label, te_data, te_label, m):
    global history
    batch_size = 1024
    epochs = 10
    metr = 'mae'
    m.compile(loss='mse', optimizer='rmsprop', metrics=[metr])
    checkpoint = ModelCheckpoint('weights.hdf5',
                                 monitor='val_mae',
                                 verbose=1,
                                 save_best_only=True)
    history = m.fit(tr_data[:], tr_label[:], batch_size=batch_size,
                    epochs=epochs, verbose=1,
                    callbacks=[checkpoint,
                               TerminateOnNaN(),
                               ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.5,
                                                 patience=2)],
                    validation_data=(te_data[:], te_label[:]),
                    shuffle=True)
    # m.load_weights(f'weights.hdf5')
    m.save(f'model')


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


def forests(tr_data, tr_label, te_data, te_label):
    clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0)
    clf.fit(tr_data, tr_label)
    print(clf.score(te_data, te_label))


def main():
    tr, te = load()
    tr_data, tr_label, te_data, te_label = take_labels(tr, te)
    # training_neuro(tr_data, tr_label, te_data, te_label, create_model(tr_data.shape[1]))
    # ploting()
    forests(tr_data, tr_label, te_data, te_label)


main()


