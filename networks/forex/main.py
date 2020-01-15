import csv
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau


def load_data(fn):
    if not os.path.isfile('data/data.npy'):
        data = []
        with open(f"data/{fn}", 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                data.append(row)

        for i in data:
            mytime = f'{i[0]} {i[1].split(".")[0]}'
            time = datetime.strptime(str(mytime), '%m/%d/%Y %H:%M:%S').timestamp()
            i[1] = time
            del i[4:], i[0]

        data = np.asarray(data).astype("float32")
        print(data[0])
        np.save('data/data', data)
    else:
        data = np.load('data/data.npy')
    return data


def norm(data):
    for i in range(len(data)):
        data[i][0] = i
    return data


def plot_graph(data):
    plt.plot(data[:, 0], data[:, 1])
    plt.plot(data[:, 0], data[:, 2])
    plt.savefig('2013.png')
    plt.show()


def preapare_data(data):
    data = data.astype("float32")
    x = []
    y = []

    lookback = 5000
    step = 500
    delay = 500
    i = lookback
    while (data.shape[0] - 1000 - i) >= 0:
        # print(i)
        x.append(np.asarray(data[(i - lookback):i, 1]))
        y.append(np.asarray(data[i:(i + delay), 1]).mean())
        i += step
    x = np.asarray(x)
    y = np.asarray(y)
    print(x.shape)
    print(y.shape)
    return x, y


def create_model(x, y, x1, y1, x2, y2):
    global history
    model = Sequential()
    # model.add(layers.LSTM(input_dim=5000, output_dim=32, return_sequences=True))
    model.add(layers.Dense(32, input_dim=5000, activation='relu'))
    model.add(layers.Dense(1))
    checkpoint = ModelCheckpoint('weights.hdf5',
                                 monitor='val_mse',
                                 verbose=1,
                                 save_best_only=True)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    history = model.fit(x, y, batch_size=32, epochs=500,
                        callbacks=[checkpoint,
                                   TerminateOnNaN(),
                                   ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.5,
                                                     patience=50)],
                        validation_data=[x1, y1])
    tmp = model.evaluate(x2, y2)
    print(tmp)
    model.save(f'model_{tmp[0]}.h5')


def ploting():
    global history
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


if __name__=="__main__":
    data = load_data('EURUSD.csv')
    data = norm(data)
    # plot_graph(data)
    x, y = preapare_data(data)
    create_model(x[:500], y[:500], x[500:700], y[500:700], x[700:], y[700:])
    ploting()





