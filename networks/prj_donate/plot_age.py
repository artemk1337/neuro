import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
from scipy.interpolate import interp1d
import pylab


public_name = ['wgcsgo', 'leagueoflegends', 'fortnite', 'dota2', 'worldofwarcraft']


arr = []
for fn in public_name:
    with open(f'data/{fn}/{fn}_age.json') as f:
        data = json.load(f)
    tmp = []
    for i in data['age'].keys():
        tmp.append(data['age'][i])
    arr.append(tmp)

print(arr)

for i in range(len(arr)):
    x = [k for k in range(14, 14 + len(arr[i]))]
    y = arr[i]
    f = interp1d(x, y, kind='linear')
    y = f(x)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='r', linewidth=3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    plt.xlim((14, 60))
    plt.ylim(0)
    plt.xlabel('age')
    plt.title(f'{public_name[i]}')
    plt.grid(which='major', color='black')
    plt.grid(which='minor', color='grey')
    plt.savefig(f'data/{public_name[i]}/{public_name[i]}_age.png')






