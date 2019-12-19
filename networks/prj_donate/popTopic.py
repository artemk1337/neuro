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


# Сохранение

def save(file_name, data):
    with open(file_name, 'w') as file:
        json.dump(data, file, separators=(',', ':'), indent=4)


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
        if len(labels) >= 5:
            break
        labels.append(key)
        items.append(item)

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(items, autopct='%1.1f%%', textprops=dict(color="w"))
    ax.legend(wedges, labels,
              title="Topics",
              loc='upper left',
              bbox_to_anchor=(-0.3, 1))
          # loc="center left",
          # bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title(f'{file_name.split("/")[1]}')
    plt.savefig(file_name)
    return True


# Сборщик

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


if __name__ == '__main__':
    files = ['dota2', 'wgcsgo', 'worldofwarcraft', 'fortnite', 'leagueoflegends']
    prefix = 'data/'
    postfix = '_groups_topics.json'
    postfix_save = '_popular_topics.json'

    for file in files:
        path = '{}{}/{}{}'.format(prefix, file, file, postfix)
        topics = popTopic(path)
        plotter('data/{}/{}_topics.jpg'.format(file, file), topics)

        if topics is not None:
            save_path = '{}{}/{}{}'.format(prefix, file, file, postfix_save)
            save(save_path, topics)
