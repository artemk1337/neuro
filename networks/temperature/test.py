import csv
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt


arr = []
with open('data/jena_climate_2009_2016.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        arr.append(np.asarray(row))

print(arr[0])
arr = np.asarray(arr[1:])
temp = np.asarray(arr[:, 2]).astype(np.float)


fig, ax = plt.subplots()
# ax.set_ylim([-30, 10])
ax.plot(temp)
plt.show()







lookback = 720
step = 6
delay = 144

mean = temp[:200000].mean(axis=0)
temp -= mean
std = temp[:200000].std(axis=0)
temp /= std

float_data = temp

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets



lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = generator(float_data, lookback=lookback, delay=delay,
                      min_index=0, max_index=200000, shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay,
                    min_index=200001, max_index=300000, step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay,
                     min_index=300001, max_index=None, step=step, batch_size=batch_size)
val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)




from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
steps_per_epoch=500,
epochs=20,
validation_data=val_gen,
validation_steps=val_steps)





import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']




