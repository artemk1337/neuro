import json


public_name = ['wgcsgo', 'leagueoflegends', 'fortnite', 'dota2', 'worldofwarcraft']


man = 0
woman = 0
for n in public_name:
    with open(f'data/{n}.json', 'r') as f:
        data = json.load(f)
    for i in data.keys():
        if int(data[i]['user']['sex']) == 2:
            man += 1
        else:
            woman += 1
    print(f'{round((woman / (man + woman)) * 100)}% - {n}')










