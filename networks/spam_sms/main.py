import numpy as np # linear algebra
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau
from keras import backend as K


def f1(y_true, y_pred):  # F-metric (with Recall and Precision)
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


num_max = 1000
max_len = 100


data = pd.read_csv('spam.csv', encoding='latin-1')
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1": 'label', "v2": 'text'})
print(data.head())
tags = data["label"]
texts = data["text"]


le = LabelEncoder()
tags = le.fit_transform(tags)
tok = Tokenizer(num_words=num_max)
tok.fit_on_texts(texts)
mat_texts = tok.texts_to_matrix(texts, mode='count')


x_train, x_test, y_train, y_test = train_test_split(texts, tags, test_size=0.3)
mat_texts_tr = tok.texts_to_matrix(x_train, mode='count')
mat_texts_tst = tok.texts_to_matrix(x_test, mode='count')


x_train = tok.texts_to_sequences(x_train)
x_test = tok.texts_to_sequences(x_test)
cnn_texts_mat = sequence.pad_sequences(x_train, maxlen=max_len)
cnn_texts_mat_tst = sequence.pad_sequences(x_test, maxlen=max_len)


def model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(num_max,)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    return model


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


def check_model(model, xtr, ytr, xts, yts):
    global history
    print(xtr.shape, ytr.shape, xts.shape, yts.shape)
    """checkpoint = ModelCheckpoint('weights.hdf5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)"""
    history = model.fit(xtr, ytr,
                        batch_size=64,
                        epochs=10,
                        callbacks=[TerminateOnNaN(),
                                   ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.5,
                                                     patience=2)],
                        verbose=1,
                        validation_split=0.3)
    """model.load_weights(f'weights.hdf5')"""
    model.save(f'model')
    print(f'Prediction: {model.evaluate(xts, yts)}')


m = model()
check_model(m, mat_texts_tr, y_train, mat_texts_tst, y_test)

# print(m.predict(mat_texts_tr[0:10]), y_train[0:10])
ploting()
