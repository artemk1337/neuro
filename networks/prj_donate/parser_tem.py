import json
# import keras
import numpy as np
import vk
import os
import operator
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib import pyplot, transforms


token = '569e2f7f569e2f7f569e2f7f3056f08a7a5569e569e2f7f0b6739a62257254d2f209689'
public_name = ['wgcsgo', 'leagueoflegends', 'fortnite', 'dota2', 'worldofwarcraft']


session = vk.Session(access_token=token)  # Авторизация
vk_api = vk.API(session)


# Собираем топики в отдельный файл
def parse_topics():
    for n in public_name:
        if not os.path.exists(f'data/{n}/{n}_groups_topics.json'):
            with open(f'data/{n}.json', 'r') as f:
                data = json.load(f)
            print(f'Create {n}.json')
            user = {}
            counter = 0
            for i in data.keys():
                # print(f'User - id {i}')
                gr = {}
                for k in data[i]['sub']['groups']['items']:
                    # print(k)
                    tmp = vk_api.groups.getById(group_id=int(k), fields=['activity'], v=5.92)[0]
                    try:
                        gr[k] = tmp['activity']
                    except:
                        pass
                user[i] = gr
                counter += 1
                print(counter)
                if counter % 50 == 0:
                    with open(f'data/{n}/{n}_groups_topics.json', 'w') as f:
                        json.dump(user, f, separators=(',', ':'), indent=4)
            print(f'Saved {n}_groups_topics.json.json')


# Собираем список самых популярных групп
def popular_groups():
    def popGroup(file_name):
        top = {}
        with open(f'data/{file_name}') as f:
            data = json.load(f)
            for k in data.keys():
                # print('current length of array -> {}'.format(len(top)))
                user = data[k]
                sub = user['sub']
                groups = sub['groups']['items']

                for group in groups:
                    result = top.get(group, None)
                    if result is None:
                        top[group] = 1
                    else:
                        top[group] += 1
        return sorted(top.items(), key=operator.itemgetter(1), reverse=True)

    def sort_dict(a):
        d = {}
        for i in range(1, 21):
            d[a[i][0]] = a[i][1]
        return d

    for f in public_name:
        print(f'Create {f}.json')
        votes = popGroup('{}.json'.format(f))
        # print(votes)
        d = sort_dict(votes)
        with open(f'data/{f}/{f}_pop_groups.json', 'w') as outfile:
            json.dump(d, outfile, separators=(',', ':'), indent=4)


# Рисуем графики
def plot_groups():
    for fn in public_name:
        with open(f'data/{fn}/{fn}_pop_groups.json') as f:
            data = json.load(f)
        info = vk_api.groups.getById(group_ids=[i for i in data.keys()], v=5.92)
        names = []
        subs = []
        for i in range(len(info)):
            names.append(info[i]['name'])
        for i in data.keys():
            subs.append(data[i])
        subs = [x / max(subs) for x in subs]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(-90)
        ax.bar(range(len(names)), subs, transform=rot + base)
        names.reverse()
        plt.yticks(range(-19, 1), names)
        plt.title(f'{fn}')
        plt.subplots_adjust(left=0.4)
        plt.savefig(f'data/{fn}/{fn}_groups.png')


# Парсим топики групп у каждого подписчика
# parse_topics()
# Парсим популярные группы
# popular_groups()
plot_groups()

