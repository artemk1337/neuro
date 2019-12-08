import numpy as np
import os
import csv
import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau
import tensorflow as tf


class Neuro(object):
    def model(self, input_shape):
        seq = Sequential()
        seq.add(Dense(500, input_dim=input_shape, activation=tf.nn.relu6))
        seq.add(Dense(50, activation=tf.nn.relu6))
        seq.add(Dense(1))
        print(seq.summary())
        return seq

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

    def train(self, x, y, x1, y1, model):
        model.compile(loss='mae', optimizer='adam', metrics=['mae'])
        try:
            model.load_weights(f'data_for_neuro/weights.hdf5')
        except:
            pass
        checkpoint = ModelCheckpoint('data_for_neuro/weights.hdf5',
                                     monitor=f'val_mae',
                                     verbose=1,
                                     save_best_only=True)
        history = model.fit(x[:], y[:],
                            batch_size=64,
                            epochs=200,
                            verbose=2,
                            callbacks=[checkpoint,
                                       TerminateOnNaN(),
                                       ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.7,
                                                         patience=10)],
                            validation_data=(x1[:], y1[:]),
                            shuffle=True)
        model.load_weights(f'data_for_neuro/weights.hdf5')
        # model.save(f'data_for_neuro/model')
        self.ploting(history)
        return model

    def prediction(self, x_te, y_te, seq):
        # print(y_te)
        # print(seq.predict(x_te).reshape(y_te.shape[0]))
        print(np.mean(np.abs(y_te - seq.predict(x_te).reshape(y_te.shape[0]))).astype('float32'), '- mae loss')

    def fit(self, x_tr, x_te, y_tr, y_te):
        if not os.path.exists('data_for_neuro/model_best'):
            seq = self.model(x_tr.shape[1])
            seq = self.train(x_tr, y_tr, x_te, y_te, seq)
            seq.save('data_for_neuro/model_best')
        else:
            seq = load_model('data_for_neuro/model_best',
                             custom_objects={'relu6': tf.nn.relu6})
        self.prediction(x_te, y_te, seq)


# Tree regression
class TR(object):
    # Constants
    def __init__(self, max_depth=5):
        # self.loss = 'mse'
        self.ft = None
        self.label = None
        self.samples = None
        self.gain = None
        self.left = None
        self.right = None
        self.threshold = None
        self.depth = 0
        self.root = None
        self.max_depth = max_depth

    def fit(self, x, y):
        # Create first level
        self.root = TR()
        self.root._grow_tree(x, y)
        self.root._prune(self.max_depth, self.root.samples)

    def predict(self, x):
        return np.array([self.root._predict(f) for f in x])

    def show_tree(self):
        self.root._show_tree(0, ' ')

    # Grow tree
    def _grow_tree(self, x, y, classes_in_top=1, depth=0):
        """
        Build a decision tree by recursively finding the best split.

        :param x:
            new X array
        :param y:
            new Y array
        :param classes_in_top:
            1 - very slow, high accuracy (acc like in sklearn); time ~ 20 sec
            500 - normal, normal accuracy
            1000+ - fast, low accuracy

        :return:
            void
        """
        self.samples = x.shape[0]
        # Stop-factor
        """ For classification: len(np.unique(y)) <= 1  -  1 class in top
        For regression: y <= 5 or more  -  5 or more classes in top
        Also can add self.depth >= self.max_depth """
        if len(np.unique(y)) <= classes_in_top:
            self.label = y[0]
            return
        self.label = np.mean(y)
        loss = self._loss_mse(y)
        best_gain, best_ft, best_th = 0.0, None, None
        for input_col in range(x.shape[1]):
            # Sort classes
            feature_level = np.unique(x[:, input_col])
            # Average between neighbors
            th = (feature_level[:-1] + feature_level[1:]) / 2
            for i in th:
                # Index 0 - left, 1 - right
                y_ = [y[x[:, input_col] <= i], y[x[:, input_col] > i]]
                new = [self._loss_mse(y_[0]), self._loss_mse(y_[1])]
                n_ = [float(y_[0].shape[0]) / self.samples, float(y_[1].shape[0]) / self.samples]
                new_gain = loss - (n_[0] * new[0] + n_[1] * new[1])
                if new_gain > best_gain:
                    best_gain, best_ft, best_th = new_gain, input_col, i
        self.ft = best_ft
        self.gain = best_gain
        self.threshold = best_th
        # Recursive
        self.left, self.right = TR(), TR()
        self.left.depth, self.right.depth = self.depth + 1, self.depth + 1
        self.left._grow_tree(x[x[:, self.ft] <= self.threshold], y[x[:, self.ft] <= self.threshold], depth + 1)
        self.right._grow_tree(x[x[:, self.ft] > self.threshold], y[x[:, self.ft] > self.threshold], depth + 1)

    def _loss_mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _prune(self, max_depth, samples):
        if self.ft == None:
            return
        self.left._prune(max_depth, samples)
        self.right._prune(max_depth, samples)
        if self.depth >= max_depth:
            self.left, self.right, self.ft = None, None, None

    def _predict(self, x):
        if self.ft != None:
            if x[self.ft] <= self.threshold:
                return self.left._predict(x)
            return self.right._predict(x)
        return self.label

    def _show_tree(self, depth, cond):
        base = '  ' * depth + cond
        if self.ft != None:
            print(str(base) + 'if X[' + str(self.ft) + '] <= ' + str(self.threshold))
            self.left._show_tree(f'{depth + 1} then ')
            self.right._show_tree(f'{depth + 1} else ')
        else:
            print(str(base) + '{value: ' + str(self.label) + ', samples: ' + str(self.samples) + '}')


def load_data(fn):
    x = []
    y = []
    with open(fn, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            x.append(np.asarray(row[0:5]))
            y.append(row[5])
    x, y = shuffle(np.asarray(x[1:]).astype('float64'),
                   np.asarray(y[1:]).astype('float64'),
                   random_state=0)
    print(x.shape)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_tr, x_te, y_tr, y_te


if __name__ == "__main__":
    x_tr, x_te, y_tr, y_te = load_data('sdss_redshift.csv')
    tm = time.time()
    rgf = TR(max_depth=10)
    # rgf = DecisionTreeRegressor(max_depth=10)
    print('<=====FITTING=====>')
    rgf.fit(x_tr, y_tr)
    print('<=====PREDICTION=====>')
    # print(y_te)
    # print(rgf.predict(x_te))
    print((np.mean(np.abs(y_te - rgf.predict(x_te)))).astype('float32'), '- mae loss')  # mae loss
    print('<=====TIME=====>')
    print(time.time() - tm, 'sec')
    # print('<=====TREE=====>')
    # rgf.show_tree()
    # quit()
    print('<=====NEURO=====>')
    neuro = Neuro()
    neuro.fit(x_tr, x_te, y_tr, y_te)
