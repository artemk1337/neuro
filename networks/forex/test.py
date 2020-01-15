import csv
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt


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

    lookback = 10000
    step = 1000
    delay = 1000
    i = lookback
    while (data.shape[0] - 1000 - i) >= 0:
        print(i)
        x.append(np.asarray(data[(i - lookback):i, 1]))
        y.append(np.asarray(data[i:(i + delay), 1]))
        i += step
    x = np.asarray(x)
    y = np.asarray(y)
    print(x.shape)
    print(y.shape)




    pass



def create_model():
    pass


if __name__=="__main__":
    data = load_data('EURUSD.csv')
    data = norm(data)
    plot_graph(data)
    preapare_data(data)





