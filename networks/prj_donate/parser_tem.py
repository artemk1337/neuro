import json
# import keras
import numpy as np
import vk
import os
import operator


token = '569e2f7f569e2f7f569e2f7f3056f08a7a5569e569e2f7f0b6739a62257254d2f209689'
public_name = ['wgcsgo', 'leagueoflegends', 'fortnite', 'dota2', 'worldofwarcraft']


session = vk.Session(access_token=token)  # Авторизация
vk_api = vk.API(session)


def parse_temes():
    def popTemGroup(fn):
        with open(f'data/{n}_v1.json', 'r') as f:
            data = json.load(f)
        arr = []
        for i in data.keys():
            for k in data[i]['sub']:
                for t in k.keys():
                    arr.append(t)
        d = {}
        for i in arr:
            tmp = d.get(i, None)
            if tmp is None:
                d[i] = 1
            else:
                d[i] += 1
        d_new = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        with open(f'data/{n}_pop_topic.json') as f:
            json.dump(d_new, f, separators=(',', ':'), indent=4)

    for n in public_name:
        if not os.path.exists(f'data/{n}.json'):
            with open(f'data/{n}.json', 'r') as f:
                data = json.load(f)

            for i in data.keys():
                gr = {}
                for k in data[i]['sub']['groups']['items']:
                    # print(k)
                    tmp = vk_api.groups.getById(group_id=int(k), fields=['activity'], v=5.92)[0]
                    gr[k] = tmp['activity']
                data[i]['sub'].clear()
                data[i]['sub'] = gr

            with open(f'data/{n}_v1.json', 'w') as f:
                json.dump(data, f, separators=(',', ':'), indent=4)
        popTemGroup(n)


def popular_groups():
    def popGroup(file_name):
        top = {}
        with open(f'data/{file_name}') as f:
            data = json.load(f)
            key = data.keys()
    
            for k in key:
                print('current length of array -> {}'.format(len(top)))
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

    for f in public_name:
        votes = popGroup('{}.json'.format(f))
        result = {'votes': votes}
        with open('data/{}_pop_groups.json'.format(f), 'w') as outfile:
            json.dump(result, outfile)


parse_temes()
popular_groups()


