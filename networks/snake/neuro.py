import pyautogui
import keras
from keras.models import Sequential
from keras.layers import Dense


class Neuro():
    def __init__(self):
        model = self.create_network()

    def create_network(self):
        model = Sequential()
        model.add(Dense(32, input_dim=4, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(4, activation='sigmoid'))
        return model



def press(x, y, x1, y1):

    pyautogui.press('down')






