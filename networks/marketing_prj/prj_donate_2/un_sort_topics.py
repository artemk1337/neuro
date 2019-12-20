import json
import operator
import matplotlib.pyplot as plt
import seaborn as sns


# Счетчик популярности категорий 

def counter(data):
    keys = data.keys()
    topics = {}

    for key in keys:
        user = data[key]
        items = user.keys()
        for item in items:
            topic = user[item]
            exsist = topics.get(topic)
            if exsist is None:
                topics[topic] = 1
            else:
                topics[topic] += 1        
    return topics


# Сортировка по популярности категорий

def sort(data):
    index = sorted(data.items(), key=operator.itemgetter(1), reverse=True)
    sorted_topic = {}

    for i in index:
        key, value = i
        if value >= 100:
            sorted_topic[key] = value
    return sorted_topic


# Отрисовка

def params(length):
    x = []
    y = []
    count = 0
    x_shift = 1000
    y_shift = 1000
    for i in range(length):
        if count == 20:
            count = 0
            x_shift = 1000
            y_shift += 1000
        x.append(x_shift)
        y.append(y_shift)
        count += 1
    return [x, y]


def func(value):
    return "{}".format(value)


def plotter(file_name, data):
    
    items = []
    labels = []

    # Top 5
    for key, item in data.items():
        if len(labels) >= 10:
            break
        labels.append(key)
        items.append(item)

    fig, ax = plt.subplots(figsize=(6, 3))
    wedges, texts, autotexts = ax.pie(items, autopct='%1.1f%%', textprops=dict(color="w"))
    ax.legend(wedges, labels,
          title="Topics",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
    plt.savefig(file_name)
    return True


def popTopic(file_name):
    with open(file_name) as file:
        try:
            data = json.load(file)
            topics = counter(data)
            topics = sort(topics)
            return topics
        except Exception as e:
            print(str(e))
            return None


files = ['hsemem', 'msuofmems']
postfix = '_groups_topics.json'

for file in files:
    topics = popTopic(f'data/{file}/{file}{postfix}')
    plotter(f'data/{file}/{file}_topics.jpg', topics)
    with open(f'data/{file}/{file}_popular_topics.json', 'w') as file:
        json.dump(topics, file)
