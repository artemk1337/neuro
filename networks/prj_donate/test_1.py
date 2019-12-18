import json
# import keras
import numpy as np
import vk
import os
import operator


token = '569e2f7f569e2f7f569e2f7f3056f08a7a5569e569e2f7f0b6739a62257254d2f209689'
public_name = ['wgcsgo', 'leagueoflegends', 'fortnite', 'dota2', 'worldofwarcraft']
public_name = ['fortnite']


session = vk.Session(access_token=token)  # Авторизация
vk_api = vk.API(session)


def parse_topics():
    def sort_dict(a):
        d = {}
        for i in range(1, 21):
            d[a[i][0]] = a[i][1]
        return d

    def popTemGroup(fn):
        with open(f'data/{fn}/{fn}_groups_topics.json', 'r') as f:
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
        d_new = sort_dict(d_new)
        with open(f'data/{fn}/{fn}_pop_topic.json') as f:
            json.dump(d_new, f, separators=(',', ':'), indent=4)

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
        popTemGroup(n)


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


parse_topics()
# popular_groups()


