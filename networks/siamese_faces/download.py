import numpy as np
import requests
import os
import time


data = np.loadtxt('facescrub_actors.txt', delimiter='\t', dtype=str)
print(data.shape)
data = data[1:]


for i in range(5405, data.shape[0]):
    print(i)
    try:
        if not os.path.exists(f'data/male/{i}_{data[i, 0]}.jpg'):
            p = requests.get(data[i, 3])
            out = open(f"data/male/{i}_{data[i, 0]}.jpg", "wb")
            out.write(p.content)
            out.close()
            print('SUCCESS')
    except:
        print('FAIL')

data = np.loadtxt('facescrub_actresses.txt', delimiter='\t', dtype=str)
print(data.shape)
data = data[1:]


for i in range(data.shape[0]):
    print(i)
    try:
        if not os.path.exists(f'data/female/{i}_{data[i, 0]}.jpg'):
            p = requests.get(data[i, 3])
            out = open(f"data/female/{i}_{data[i, 0]}.jpg", "wb")
            out.write(p.content)
            out.close()
            print('SUCCESS')
    except:
        pass


