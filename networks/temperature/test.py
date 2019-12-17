import csv
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt


arr = []
with open('jena_climate_2009_2016.csv', 'r') as f:
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








