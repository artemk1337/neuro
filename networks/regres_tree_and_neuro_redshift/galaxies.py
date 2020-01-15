import forest
import warnings
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def load_data(fn):
    x = []
    y = []
    with open(fn, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            top = np.asarray(row)
            break
        for row in csv_reader:
            x.append(np.asarray(row[0:5]))
            y.append(row[5])
    x, y = shuffle(np.asarray(x).astype('float32'),
                   np.asarray(y).astype('float32'),
                   random_state=0)
    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.2, random_state=0)
    del x, y
    return x1, x2, y1, y2, top


def load_data_predict(fn):
    x3 = []
    with open(fn, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            x3.append(np.asarray(row[0:5]))
    x3 = np.asarray(x3[1:]).astype('float32')
    return x3


def save(fn, y1, prediction_1, y2, prediction_2):
    text = {"train": np.std(y1 - prediction_1).astype('float64'),
            "test": np.std(y2 - prediction_2).astype('float64')}
    print(text)
    with open(fn, 'w') as f:
        json.dump(text, f, indent=4, separators=(',', ': '))


def plot():
    plt.figure()
    plt.scatter(train_y, rgf.predict(train_x), color='palegoldenrod', )
    plt.scatter(test_y, rgf.predict(test_x), color='green', alpha=0.3)
    plt.xlabel('expected')
    plt.ylabel('prediction')
    plt.title('Tree prediction')
    x1 = np.linspace(0, train_y.max())
    y1 = np.linspace(0, test_y.max())
    plt.plot(x1, y1, color='olive')
    plt.show()
    plt.savefig('redshift.png')


def save_csv(x3, prediction_3, top):
    res = []
    for i in range(x3.shape[0]):
        res.append(np.asarray([*x3[i], prediction_3[i]]))
    res = np.asarray(res)
    res = np.concatenate((np.asarray([top]), res))

    with open('sdss_predict.csv', "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for line in res:
            writer.writerow(line)


warnings.filterwarnings("ignore", category=RuntimeWarning)
x1, x2, y1, y2, top = load_data('sdss_redshift.csv')
x3 = load_data_predict('sdss.csv')
prediction_1, prediction_2, prediction_3 = forest.RandomForest(trees=50, max_depth=10)\
                                                 .fit_and_predict(x1, x2, y1, y2, x3)
plot(y1, y2, prediction_1, prediction_2)
save('redhsift.json', y1, prediction_1, y2, prediction_2)
save_csv(x3, prediction_3, top)


