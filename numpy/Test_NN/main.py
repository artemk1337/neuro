from math import exp
import numpy as np


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_derivative(x):
    return (exp(-x)) / ((1 + exp(-x)) ** 2)


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return np.where(x <= 0, 0, 1)


class NeuralNetwork:
    def ___init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weightsl
        d_weights2 = np.dot(self.layer1.T,
                            (2 * (self.y - self.output)
                             * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,
                            (np.dot(
                                2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                self.weights2.T)
                             * sigmoid_derivative(self.layer1)))
        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == '__main__':
    x = np.random.uniform(-1, 1, (3, 3))
    print(x)
    print(relu(x))


