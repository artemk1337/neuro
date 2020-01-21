import numpy as np
import os
import csv
import time
import matplotlib.pyplot as plt
from scipy import optimize
# from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# Tree regression
class TR(object):
    # Constants
    def __init__(self, max_depth=5):
        # self.loss = 'mse'
        self.ft = None
        self.left = None
        self.right = None
        self.label = None
        self.samples = None
        self.gain = None
        self.threshold = None
        self.depth = 0
        self.start = None
        self.max_depth = max_depth

    def fit(self, x, y):
        """
        :param x:
            X - array
        :param y:
            Y - array
        """
        self.start = TR(self.max_depth)
        self.start._grow_tree_(x, y)

    def predict(self, x):
        return np.array([self.start._predict_(f) for f in x])

    def show_tree(self):
        self.start._show_tree_(0, ' ')

    def _grow_tree_(self, x, y):
        """
        Build a decision tree by recursively finding the best split.

        :param x:
            new X array
        :param y:
            new Y array

        :return:
            void
        """
        self.samples = x.shape[0]
        self.label = np.mean(y)
        self.ft = None
        self.gain = np.inf
        self.threshold = x[0, 0]
        # Stop-factor
        """ For classification: len(np.unique(y)) <= 1  -  1 class in top
        For regression: y <= 5 or more  -  5 or more the nearest classes in top """
        if len(np.unique(y)) <= 10 or self.depth >= self.max_depth:
            return
        # man - https://spark.apache.org/docs/2.2.0/mllib-decision-tree.html
        self._optimizer_(x, y)
        # Recursive
        self.left, self.right = TR(), TR()
        self.left.depth, self.right.depth = self.depth + 1, self.depth + 1
        self.left.max_depth, self.right.max_depth = self.max_depth, self.max_depth
        self.left._grow_tree_(x[x[:, self.ft] <= self.threshold], y[x[:, self.ft] <= self.threshold])
        self.right._grow_tree_(x[x[:, self.ft] > self.threshold], y[x[:, self.ft] > self.threshold])

    # Find best splitter
    def _optimizer_(self, x, y):
        best_ft, best_th, best_gain = self.ft, self.threshold, self.gain
        for col in range(x.shape[1]):
            # Mae like in sklearn, works VERY slow
            """# Sort classes
            feature_level = np.unique(x[:, col])
            # Average between neighbors
            th = (feature_level[:-1] + feature_level[1:]) / 2
            for i in th:
                # Index 0 - left, 1 - right
                y_ = [y[x[:, col] <= i], y[x[:, col] > i]]
                new = [self._loss_mse_(y_[0]), self._loss_mse_(y_[1])]
                k = [float(y_[0].shape[0]) / self.samples, float(y_[1].shape[0]) / self.samples]
                new_gain = k[0] * new[0] + k[1] * new[1]
                if new_gain < best_gain:
                    best_gain, best_ft, best_th = new_gain, col, i"""
            # loss good, works slow, correct!
            feature_level = np.unique(x[:, col])
            res = optimize.minimize_scalar(self._minimize_scalar_,
                                           args=(col, x, y),
                                           bounds=(feature_level[1],
                                                   feature_level[-1]),
                                           method='Bounded')
            if res.fun < best_gain:
                best_gain, best_ft, best_th = res.fun, col, res.x
            # Mean digit, works fast and better!!!
            '''i = np.mean(x[:, col])
            y_ = [y[x[:, col] <= i], y[x[:, col] > i]]
            new = [self._loss_mse_(y_[0]), self._loss_mse_(y_[1])]
            k = [float(y_[0].shape[0]) / self.samples, float(y_[1].shape[0]) / self.samples]
            new_gain = k[0] * new[0] + k[1] * new[1]
            if new_gain < best_gain:
                best_gain, best_ft, best_th = new_gain, col, i'''
        self.ft = best_ft
        self.gain = best_gain
        self.threshold = best_th

    # For method from seminar
    def _minimize_scalar_(self, value, feature, x, y):
        y_ = [y[x[:, feature] <= value], y[x[:, feature] > value]]
        new = [self._loss_mse_(y_[0]), self._loss_mse_(y_[1])]
        k = [float(y_[0].shape[0]) / self.samples, float(y_[1].shape[0]) / self.samples]
        return k[0] * new[0] + k[1] * new[1]

    def _loss_mse_(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _predict_(self, x):
        if self.ft != None:
            if x[self.ft] <= self.threshold:
                return self.left._predict_(x)
            return self.right._predict_(x)
        return self.label

    def _show_tree_(self, depth, separator):
        base = ' ' * 2 * depth + separator
        if self.ft != None:
            print(str(base) + 'if X[' + str(self.ft) + '] <= ' + str(self.threshold))
            self.left._show_tree_(depth + 1, f'{depth + 1} then ')
            self.right._show_tree_(depth + 1, f'{depth + 1} else ')
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
    print((np.mean(np.abs(y_te - rgf.predict(x_te)))).astype('float32'), '- mae loss')  # mae loss
    print(np.std(y_te - rgf.predict(x_te)), '- std loss')  # std loss
    print('<=====TIME=====>')
    print(time.time() - tm, 'sec')
    # print('<=====TREE=====>')
    # rgf.show_tree()
    print('<=====NEURO=====>')
    neuro = Neuro()
    neuro.fit(x_tr, x_te, y_tr, y_te)
