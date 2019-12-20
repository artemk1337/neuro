import datetime
import json
import vk
import operator
import matplotlib.pyplot as plt
import os
import gensim
import gensim.corpora as corpora
import pyLDAvis
import nltk
from nltk.corpus import stopwords
import pyLDAvis.gensim
import pymorphy2
from wordcloud import WordCloud
import numpy as np
import operator
from PIL import Image


public_name = ['hsemem', 'msuofmems']


for name in public_name:
    with open(f'data/{name}.json', 'r') as file:
        data = json.load(file)
    city = {}
    for i in data.keys():
        try:
            town = data[i]['user']['home_town'].lower()
            if ' ' in town:
                town = town.splt(' ')[1]
            if town == 'moscow':
                town = 'москва'
            if town in city.keys() and town != '':
                city[town] += 1
            elif town != '':
                city[town] = 1
        except:
            pass
    a = sorted(city.items(), key=operator.itemgetter(1), reverse=True)
    aaa = []
    for i in range(len(a)):
        if i < 20:
            aaa.append(a[i][1])
        else:
            break
    names = []
    for i in range(len(a)):
        if i < 20:
            names.append(a[i][0])
        else:
            break
    total = sum(aaa)
    labels = [f"{n} ({v / total:.1%})" for n, v in zip(names, aaa)]
    fig, ax = plt.subplots()
    ax.pie(aaa)
    ax.legend(labels,
              title="Topics",
              loc='upper left',
              bbox_to_anchor=(-0.4, 1.15),
              fontsize='small',
              framealpha=0.3)
    plt.title(f'{name}')
    plt.show()


