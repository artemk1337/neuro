#import libaries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt

#read data.
data = pd.read_csv('data/heart.csv')

#pre-process categorical features
catagorialList = ['sex','cp','fbs','restecg','exang','ca','thal']
for item in catagorialList:
    data[item] = data[item].astype('object')
print(data.shape)
data = pd.get_dummies(data, drop_first=True)

#normalize training features
y = data['target'].values
y = y.reshape(y.shape[0],1)
x = data.drop(['target'],axis=1)
minx = np.min(x)
maxx = np.max(x)
x = (x - minx) / (maxx - minx)
print(x.shape)
x.head()

#split training set and test set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

print(x_train.shape)

#set up neural network. Structure is 21->12->1
model = Sequential()
model.add(Dense(12, input_dim=21, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

#Compile and train neural network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=500, batch_size=x_train.shape[0])
scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def ploting():
    # print(history.history.keys())
    loss = history.history['loss']
    acc = history.history['acc']
    epochs = range(1, len(loss) + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(epochs, loss, 'bo', label='Training loss')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    for ax in fig.axes:
        ax.grid(True)
    plt.savefig('graph')
    plt.show()


ploting()