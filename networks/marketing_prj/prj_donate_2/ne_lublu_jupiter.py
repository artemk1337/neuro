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
public_name = ['msuofmems']

token = 'fcce1d69fcce1d69fcce1d690cfca0ac80ffccefcce1d69a132d178f01db5971a49e0e5'  # Сервисный ключ доступа
# fcce1d69fcce1d69fcce1d690cfca0ac80ffccefcce1d69a132d178f01db5971a49e0e5

session = vk.Session(access_token=token)  # Авторизация
vk_api = vk.API(session)
print('Success connect!')


def ids():
    for k in public_name:
        first = vk_api.groups.getMembers(group_id=k, v=5.92)
        data = first["items"]
        for i in range(1, first["count"] // 1000 + 1):
            print(i)
            data += vk_api.groups.getMembers(group_id=k, v=5.92, offset=i * 1000)["items"]
        with open(f'data/{k}.txt', "w") as file:
            for item in data:
                file.write(f'{item}\n')


# Собираю все страницы пользователей вк
# Сохраняю в файл чтобы не потерять
def parse_pages():
    for name in public_name:
        peop = 0
        with open(f'data/{name}.txt', 'r') as file:
            data = file.readlines()
        massive = [int(i.split('\n')[0]) for i in data]
        res = {}
        for i in range(0, len(massive), 1):
            print('People', peop)
            try:
                user = vk_api.users.get(user_ids=massive[i], v=5.92, fields=['sex', 'bdate',
                                                                             'city', 'country', 'home_town'])[0]
                print(user)
                if len(user['bdate'].split('.')) == 3:
                    res[massive[i]] = {'user': user,
                                       'wall': vk_api.wall.get(owner_id=user['id'],
                                                               v=5.92),
                                       'sub': vk_api.users.getSubscriptions(user_id=massive[i],
                                                                            v=5.92)}
                    peop += 1
                    if peop % 100 == 0:
                        with open(f'data/{name}.json', 'w') as file:
                            json.dump(res, file)
            except Exception as e:
                print(e)
                pass


# Узнаю возраст подписчиков
def years_old():
    # Рисую графики, зависимость количества подписчкиов от их возраста
    def plot():
        arr = []
        for fn in public_name:
            with open(f'data/{fn}/{fn}-ages.json', 'r') as file:
                data = json.load(file)
            arr1 = []
            for i in data['age'].keys():
                arr1.append(data['age'][i])
            arr.append(arr1)
        for i in range(len(arr)):
            x = [t for t in range(14, 14 + len(arr[i]))]
            plt.plot(x, arr[i], linewidth=3)
            plt.xlabel('Возраст')
            plt.title(f'{public_name[i]}')
            plt.grid(which='major')
            plt.xticks([k for k in range(14, 101, 3)])
            plt.savefig(f'data/{public_name[i]}/{public_name[i]}-ages.jpg')
            plt.show()

    # Прогоняю два паблика.
    # Ищу средний возраст и отдельно для каждого подписчика (от 14 до 100 лет) (14 - мин. возраст)
    for f in public_name:
        mean = 0
        with open(f'data/{f}.json', 'r') as file:
            born = json.load(file)
        ages = {}
        for i in range(14, 101):
            ages[i] = 0
        for i in born.keys():
            difference = datetime.datetime.today().year - \
                         datetime.datetime.strptime(born[i]['user']['bdate'], "%d.%m.%Y").year
            mean += difference
            if 14 <= difference < 101:
                ages[difference] += 1
        mean = int(mean / len(born))
        with open(f'data/{f}/{f}-ages.json', 'w') as file:
            json.dump({'mean': mean, 'age': ages}, file)
    # Отрисовываю графики
    plot()


def groups():
    # Собираю группы, на которые подписаны пользователи
    def parse_groups():
        for file_name in public_name:
            k = 0
            with open(f'data/{file_name}.json', 'r') as f:
                data = json.load(f)
                user = {}
                for i in data.keys():
                    print(k)
                    id_proups = {f"{data[i]['sub']['groups']['items'][k]}": data[i]['sub']['groups']['items'][k]
                                 for k in range(len(data[i]['sub']['groups']['items']))}
                    user[i] = id_proups
                    k += 1
            with open(f'data/{file_name}/{file_name}-group-list.json', 'w') as file:
                json.dump(user, file)

    # Рисуем картинку с популярными группами
    def popular_gr():
        for file_name in public_name:
            with open(f'data/{file_name}/{file_name}-group-list.json', 'r') as file:
                data = json.load(file)
            gr = {}
            for i in data.keys():
                for k in data[i]:
                    if k in gr.keys():
                        gr[k] += 1
                    else:
                        gr[k] = 1
            a = sorted(gr.items(), key=operator.itemgetter(1), reverse=True)
            aaa = []
            for i in range(len(a)):
                if i < 10:
                    aaa.append(a[i][1])
                else:
                    break
            names = []
            for i in range(len(a)):
                if i < 10:
                    names.append(vk_api.groups.getById(group_ids=int(a[i][0]), v=5.92)[0]['name'])
                else:
                    break
            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(aaa, autopct='%1.1f%%', textprops=dict(color="w"))
            ax.legend(wedges, names,
                      title="Topics",
                      loc='upper left',
                      bbox_to_anchor=(-0.4, 1.15),
                      fontsize='small',
                      framealpha=0.3)
            plt.title(f'{file_name}')
            plt.savefig(f'data/{file_name}/{file_name}-group.png')
            plt.show()

    parse_groups()
    popular_gr()


# Определяем, из каких городов прибыли студенты
def home_town():
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


"""
bad_w = 'твой com наш свой каждый http https vk id ваш её который wall'
part_of_speach = ['INTJ', 'PRCL', 'NPRO', 'CONJ', 'PREP', 'COMP', 'ADVB']


top_w=10


text = np.load('all-repost.npy')

# Скачиваем стоп-слова
nltk.download('stopwords')
stopwords_ru = stopwords.words('russian')
stopwords_en = stopwords.words('english')

# Токенизируем
text = [gensim.utils.simple_preprocess(i) for i in text]

# Количество слов в репосте
text = [x for x in text if len(x) > 20]

# Удаляет плохие словечки
morph = pymorphy2.MorphAnalyzer()

# Начальная форма
text = [[morph.parse(word)[0].normal_form for word in i] for i in text]

# Удаляем части речи
text = [[word for word in words if morph.parse(word)[0].tag.POS not in part_of_speach] for words in text]

# Удаляю слова
text = [[word for word in x if word not in stopwords_ru] for x in text]
text = [[word for word in x if word not in stopwords_en] for x in text]
text = [[word for word in x if word not in bad_w.split(' ')] for x in text]

# Создаем и сразу же фильтруем словарь
dict = corpora.Dictionary(text)
dict.filter_extremes(no_below=10, no_above=0.5)
corpus = [dict.doc2bow(i) for i in text]

# Создаю LDA модель для анализа
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dict, num_topics=top_w)

# Если нет директории, то создадим
if not os.path.isdir('LDA'):
    os.mkdir('LDA')
# Созраним модель, чтобы в будущем использовать
lda_model.save('LDA/LDA_model_repost')

# Визуализируем
pyLDAvis.enable_notebook()  # Для юпитер ноутбука
visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, dict)
pyLDAvis.save_html(visualisation, f'LDA/Visual-repost.html')

# Берем самые популярные топики по 50 слов в каждом
topics = lda_model.show_topics(num_topics=top_w, num_words=50, formatted=False)
np.save('all_words_repost', topics)

"""

text = np.load('all_words_repost.npy', allow_pickle=True)
# text = topics

"""
# Соберем все сова по вероятности в одну строку
popular = {}
for category in text:
    words = category[1]
    for i in range(len(words)):
        if i > 50:
            break
        popular.setdefault(words[i][0], words[i][1])
sorted_words = sorted(popular.items(), key=operator.itemgetter(1), reverse=True)
text = ''
for item in sorted_words:
    text += ' ' + item[0]

wc = WordCloud(max_font_size=200, max_words=150, width=1500, height=1500, background_color="white")
wc.generate(text)
a = wc.to_array()
a = Image.fromarray(a)
a.show()
"""


# ids()
parse_pages()
# years_old()
# groups()
# home_town()
