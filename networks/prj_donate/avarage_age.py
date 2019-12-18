import json
import datetime


public_name = ['wgcsgo', 'leagueoflegends', 'fortnite', 'dota2', 'worldofwarcraft']


def age(file_name):
    ages = {}
    mean_age = 0

    for i in range(14, 61):
        ages[i] = 0

    with open(file_name) as f:
        data = json.load(f)
        key = data.keys()

    for k in key:
        item = data[k]
        user = item['user']
        date = user['bdate']
        today = datetime.datetime.today()
        bdate = datetime.datetime.strptime(date, "%d.%m.%Y")

        diff = today.year - bdate.year
        mean_age += diff

        if 14 <= diff <= 60:
            ages[diff] += 1

    mean_age /= len(data)
    mean_age = int(mean_age)
    return [ages, mean_age]


for f in public_name:
    ages, mean_age = age(f'data/{f}.json')
    result = {'age': ages, 'mean_age': mean_age}

    with open(f'data/{f}/{f}_age.json', 'w') as outfile:
        json.dump(result, outfile, separators=(',', ':'), indent=4)