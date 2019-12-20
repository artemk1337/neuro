import datetime
import json
import vk
import operator
import matplotlib.pyplot as plt


token = '45708ab345708ab345708ab3c0451e2fb64457045708ab3188bf174880fe2406a7e83d6'  # Сервисный ключ доступа

public_name = ['hsemem', 'msuofmems']

session = vk.Session(access_token=token)  # Авторизация
vk_api = vk.API(session)


def home_town():
    for name in public_name:
        with open(f'data/{name}.json', 'r') as f:
            data = json.load(f)
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
        plt.savefig(f'data/{name}/{name}-home_town.png')
        plt.show()


home_town()

