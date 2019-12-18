#!/usr/bin/env python
# coding: utf-8


from pymystem3 import Mystem

m = Mystem()  # обхявили предобработчик

lemmas = m.lemmatize(texts[-1])  # натравили его на текст
print(lemmas[:50])


# объявили предобработчик с опцией выкидывать пробелы и пунктуацию
m = Mystem(entire_input=False)
lemmas = m.lemmatize(texts[-1])
print(lemmas[:50])


# Запустим предобработчик на всех данных. Это займёт какое-то время.


get_ipython().run_cell_magic('time', '', 'texts_prep = [m.lemmatize(text) for text in texts]\ntexts_prep[-1][:50]')


# Дополнительно прогоним все тексты через список стоп-слов, чтобы выбросить всякий хлам. Если на компьютере нет nltk, его придётся установить командой `!pip install nltk`. Внутри лежит куча разных полезных вещей, которые обычно применяютс для анализа текстов. 

# In[ ]:


# нужно только после первой установки, чтобы скачать стоп-слова на свой комп :) 
# в nltk много других вещей, которые можно также скачать
# nltk.download("stopwords") 


# In[16]:


from nltk.corpus import stopwords

# Например, там лежат стоп-слова 
stopwords_ru = stopwords.words('russian') 
stopwords_ru[:10]


# Очищаем каждый текст от стоп-слов.

# In[17]:


texts_prep = [[word for word in text if word not in stopwords_ru] for text in texts_prep]


# После предобработки лучше данные сохранить на компьютер. Делать это вы уже умеете. 

# In[18]:


with open('wall_prep.pickle', 'wb') as f:
    pickle.dump(texts_prep, f)


# # 4. Латент Дирихле Алокатион (LDA)  модель
# 
# Вот тут лежит лучший друг человека, [документация](https://radimrehurek.com/gensim/models/ldamodel.html) LDA в gensim.
# 
# ## 4.1 Словари и чистка от стоп-слов
# 
# Итак, для начала попробуем воспользоваться для кластеризации групп по контенту библиотекой `gensim`, внутри которой лежит довольно популярный для этих целей алгоритм под названием LDA. Если у вас нет библиотеки `gensim`, установите её: `!pip install gensim`.

# In[19]:


from gensim import corpora, models


# Для его работы коллекция документов должна быть представлена в виде списка списков, каждый внутренний список соответствует отдельному документу и состоит из его слов. Пример коллекции из двух документов: 
# 
# ```[["hello", "world"], ["programming", "in", "python"]]```
# 
# Преобразуем наши данные в такой формат, а затем создадим объекты corpus и dictionary, с которыми будет работать модель.

# In[20]:


dictionary = corpora.Dictionary(texts_prep)                 # составляем словарь из терминов
corpus = [dictionary.doc2bow(text) for text in texts_prep]  # составляем корпус документов


# В векторе `texts_prep` лежит лемматизированный контент из группы. В векторе `corpus` лежит то сколько раз какое слово встречается в векторе texts. Например, нулевое слово встречается 1 раз. Соответствие каждого индекса конкретному слову лежит в словаре `dictionary`.

# In[21]:


print(texts_prep[0][:12])
print(corpus[0][:12])


# Можно посмотреть какое слово какой цифрой закодировано и наоборот. Также можно посмотреть на длину словаря. 

# In[22]:


dictionary.token2id['команда']


# In[23]:


len(dictionary)


# Интересно было бы узнать какие слова встречаются в группах очень часто, а какие очень редко. Для того, чтобы узнать это, построим красивую картиночку. Это сделает чудо-функция ниже. Не бойтесь такого кодища, она простая!

# In[24]:


# Посмотрим как часто какие слова встречаются в наших текстах 
from collections import defaultdict
import itertools    


# Команда для строительства графика.
def word_freq_plot(dictionary,corpus, k2=100, k1=0):
    # Создаём по всем текстам словарик с частотами
    total_word_count = defaultdict(int)
    for word_id, word_count in itertools.chain.from_iterable(corpus):
        total_word_count[dictionary.get(word_id)] += word_count

    # Сортируем словарик по частотам
    sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)     
    
    # Делаем срез и по этому срезу строим картиночку 
    example_list = sorted_word_count[k1:k2]
    word = []
    frequency = []
    for i in range(len(example_list)):
        word.append(example_list[i][0])
        frequency.append(example_list[i][1])

    indices = np.arange(len(example_list))
    
    plt.figure(figsize=(22,10))
    plt.bar(indices, frequency)
    plt.xticks(indices, word, rotation='vertical',fontsize=12)
    plt.tight_layout()
    pass


print('Размер словаря до фильтрации: {}'.format(len(dictionary)))
word_freq_plot(dictionary, corpus)

# можно сохранить картинку на компьютер для презентации или красоты dpi - её качество
# plt.savefig('Text/images/wordcount.png', dpi=500)


# Повсюду одни сплошные предлоги и частицы с местоимениями. Эти слова часто встречаются в разных текстах и не несут абсолютно никакой смысловой нагрузки. Как уже говорилось ранее, такие слова называют стоп-словами. Если не выкинуть их из модели, они всплывут в каждой теме и всё зашумят. Можете попробовать на досуге построить с ними свою модель. Для построения адекватной модели от них необходимо избавиться.
# 
# Для борьбы с такими словами используется два подхода. В первом подходе такие слова зачищают с помощью специального списка из стоп-слов. Во втором подходе словарь фильтруют по частоте. Обычно его фильтруют с двух сторон: убирают очень редкие слова (в целях экономии памяти, а также увеличения числа наблюдений для оценки каждого коэффициента) и очень частые слова (в целях повышения интерпретируемости тем).
# 
# Отфильтруем словарь по частоте и избавимся от всяких часто употребляемых стоп-слов. Заодно выбросим очень редкие слова, так как по ним ничего нельзя определенно сказать, а переобчаться под них мы не хотим.

# In[26]:


# слово должно встретиться хотябы 10 раз и не более чем в 50% документов
dictionary.filter_extremes(no_below=10, no_above=0.5)

print('Размер словаря после фильтрации: {}'.format(len(dictionary)))
word_freq_plot(dictionary, corpus)


# In[27]:


# составляем корпус документов по отфиьлтрованному словарю
corpus = [dictionary.doc2bow(text) for text in texts_prep]  


# ## 4.2 Наша первая LDA модель. 
# 
# Обучим модель с $20$ тематиками. Обычно непонятно сколько тематик надо выбрать. Пробуют разное число на неочень больших выборках, и смотрят насколько хорошо получилось либо с точки зрения разных сложных метрик, либо с точки зрения здравого смысла. Мы будем пользоваться вторым. Надеюсь, что у нас он есть.
# 
# Учить будем [распаралеленный вариант модели,](https://radimrehurek.com/gensim/models/ldamulticore.html) чтобы побыстрее выучилась. ;) 

# In[28]:


get_ipython().run_cell_magic('time', '', '\nT = 20\n\nlda_model =  models.ldamulticore.LdaMulticore(corpus=corpus,       # корпус для обучения\n                                             id2word=dictionary,   # словарь для обучения\n                                             num_topics=T,         # число тематик, которые мы выделяем\n                                             random_state=42,      # чтобы воспроизводилось при перезапуске\n                                             passes=20)            # число проходов по коллекции документов \n                                                                   # (если долго работает, уменьшите его!)')


# Сохраним модель.  Она училась долго. Негоже ей пропадать. 

# In[29]:


get_ipython().system('mkdir lda_model')


# In[30]:


# Сохранение модели
model_path = 'lda_model/'

lda_model.save(model_path + "ldamodel")
np.save(model_path + 'explog', lda_model .expElogbeta)


# In[31]:


get_ipython().system('ls -lh lda_model/')


# Теперь мы можем посмотреть для каждого документа распределение по темам. Например, для первого документа оно выглядит вот так: 

# In[32]:


lda_model.get_document_topics(corpus[299])


# In[33]:


print(texts_prep[299][:50])


# Вопрос только в том, что эти тематики обозначают. 

# # 5. Интерпретация тематик

# In[34]:


# Загрузка модели
ldamodel = models.ldamodel.LdaModel.load(model_path + "ldamodel")

expElogbeta = np.load(model_path + 'ldamodel.expElogbeta.npy')
ldamodel.expElogbeta = expElogbeta

T = ldamodel.num_topics # число тематик 
T


# In[35]:


ldamodel.get_document_topics(corpus[99])


# In[36]:


topics = ldamodel.show_topics(num_topics=T, num_words=20, formatted=False)

for i,top in enumerate(topics):
    print(i, [item[0] for item in top[1]],'\n')


# In[96]:


themes_names = {
     0: 'скидки цены',
     1: 'добрая тема',
     2: 'рецепты',
     3: 'новые релизы',
     4: '???_1',
     5: 'учёба',
     6: '???_2',
     7: 'спортивные победы',
     8: 'конкурсы',   
     9: '???_3',
     10: 'конкурсы',
     11: 'музыка',
     12: '???_4',
     13: 'уход за собой',
     14: '???_5',
     15: 'приюты помощь',
     16: '???_6',
     17: 'кино',
     18: 'ML и python',
     19: '???_7'
 }

# заглушка, если мне влом будет всё проинтерпретировать
#themes_names = {i:'cluster_{}'.format(i) for i in range(T)}


# Построим для каждой темы облака из слов (мы так уже делали в прошлых семинарах, когда говорили про тексты. 

# In[38]:


from wordcloud import WordCloud  # Пакет для построения облаков слов

def getTopicWords(topic_number, lda_30_topics, scaling=10000):
    """
        Возвращает склеенные в текст слова топика, отмасштабированные по вероятности
    """
    # забираем слова и вероятности из топиков
    topic = lda_30_topics[topic_number][1]
    # инициализируем пустой лист для хранения текста
    topic_multiplied = []
    # проходимся по всем словам топика
    for word, prob in topic:
        # повторяем слово N раз, где N = int(word_probability * scaling)
        topic_multiplied.append([word] * int(prob*scaling))
    
    # склеиваем все слова в один текст
    topic_multiplied = sum(topic_multiplied, [])
    topic_multiplied = ",".join(topic_multiplied)
    
    return topic_multiplied


# Достали топики.

# In[39]:


topics = ldamodel.show_topics(num_topics=T, num_words=500, formatted=False)


# Строим картинки!

# In[136]:


wordcloud = WordCloud(background_color="white", max_words=2000, width=900, height=900, collocations=False)

wordcloud = wordcloud.generate(getTopicWords(7, topics, scaling=10000))

plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off");


# Если хочется, можно построить сразу все картинки :)

# In[52]:


fig = plt.figure(figsize=(16, 20))
for i in range(T):
    ax = fig.add_subplot(5, 4, i+1)
    ax.grid(False)
    ax.axis('off')

    wordcloud = WordCloud(background_color="white", max_words=2000, width=900, height=900, collocations=False)
    wc = wordcloud.generate(getTopicWords(i, topics, scaling=10000))
    
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title("{}".format(themes_names[i]))
    
# сохраняем картинку для презы или ещё чего, если надо
# plt.savefig('clusters.png', dpi=450)


# Корреляция между тематиками. 

# In[78]:


# матрица тем
Phi = ldamodel.state.get_lambda()
print(Phi.shape)

S = pd.DataFrame(Phi.T).corr()


# In[79]:


sns.set(font_scale=1.3)
plt.subplots(figsize=(14, 14))
sns.heatmap(S, square=True,
            annot=True, fmt=".2f", linewidths=0.1, cmap="RdYlGn_r",
            yticklabels=list(themes_names.values()), 
            xticklabels=list(themes_names.values()),
            cbar=False);


# # 6. Визуализация тематик
# 
# С помощью специального пакета можно построить для анализа тематик небольшую визуализацию. Если модель тяжёлая, она будет строиться долго. Если пакета нет, можно установить его, запустив в ячейке `pip install pyLDAvis`.

# In[41]:


import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary=ldamodel.id2word)
vis


# # 7. Сохраняем таблички c тематическими профилями 
# 
# Сделаем и сохраним матрицу встречаемости тем на разных стенах. Это наш основной продукт. 

# In[86]:


get_ipython().run_cell_magic('time', '', 'th_df = [ ]  # матрица документы на темы \n\nfor item in corpus:\n    dct = dict(zip(range(T), [0]*T))\n    dct.update(dict(ldamodel.get_document_topics(item)))\n    th_df.append(dct)\n    \nth_df = pd.DataFrame(th_df)  ')


# In[97]:


th_df.columns = list(themes_names.values())
th_df.shape


# In[98]:


th_df.head()


# Самые частые темы по максимальной встречаемости.

# In[99]:


from collections import Counter
top_th = Counter([themes_names[item] for item in th_df.get_values().argmax(axis=1)])
top_th.most_common()


# Допишем в табличку названия групп. __Если бы мы скачали для каждого пользователя какие-нибудь социаль-демографические характеристики, например, пол, или города или ещё что-то, мы могли бы их тоже дописать в эту таблицу. Это могло бы нам помочь посмотреть какие именно тематики всплывают по разным соц-дем категориям. Возможно,там были бы какие-то интересные закономерности.__  Не побрезгуйте в домашках соц-демом и что-нибудь скачайте по нему.

# In[100]:


th_df['group'] = group
th_df.head()


# In[112]:


th_df.to_csv('theme_data.tsv', sep='\t', index=False) # сохранили данные


# # 8. Тематические профили подписчиков по группам.

# In[130]:


th_df = pd.read_csv('theme_data.tsv', sep='\t')  # считали данные
th_df.drop('Unnamed: 0', axis=1, inplace=True) # КРИВО СОХРАНИЛИ ААААА
th_df.head()


# Сделаем группировку по всем темам и посмотрим как именно они выглядят по подписчикам.

# In[131]:


groups_themes = th_df.groupby('group').mean()
groups_themes


# In[132]:


sns.set(font_scale=1.3)
plt.subplots(figsize=(14, 14))

sns.heatmap(groups_themes, square=True,
            annot=True, fmt=".2f", linewidths=0.1, cmap="RdYlGn_r",
            yticklabels=list(groups_themes.index), 
            xticklabels=list(groups_themes.columns),
            cbar=False);


# Для красоты картинки занулим все маргинальные темы.

# In[133]:


groups_themes[groups_themes < 0.03] = 0


# In[134]:


sns.set(font_scale=1.3)
plt.subplots(figsize=(14, 14))

sns.heatmap(groups_themes, square=True,
            annot=True, fmt=".2f", linewidths=0.1, cmap="RdYlGn_r",
            yticklabels=list(groups_themes.index), 
            xticklabels=list(groups_themes.columns),
            cbar=False);


# Если бы мы скачали из контакта побольше данных по соц-дему, можно было бы посмотреть и на другие группировки!

# # 9. Похожесть групп между собой.
# 
# Похожесть групп будем мерять косинусным расстоянием! 

# In[138]:


from scipy.spatial.distance import cosine

n = groups_themes.shape[0]  # выясняем число групп
R = np.zeros((n,n))         # заводим матрицу расстояний 


for i in range(n):
    vect1 = groups_themes.iloc[i]     # выделяем вектор для первого 
    for j in range(n):
        vect2 = groups_themes.iloc[j] # выделяем второй вектор 
        R[i,j] = cosine(vect1, vect2) # ищем косинусное расстояние


# Чем больше коминус, тем сильнее группы не похожи друг на друга по своим подписчикам!

# In[140]:


sns.set(font_scale=1.5)
plt.subplots(figsize=(8, 8))
sns.heatmap(R, square=True,
            annot=True, fmt=".2f", linewidths=0.1, cmap="RdYlGn_r",
            yticklabels=list(groups_themes.index), 
            xticklabels=list(groups_themes.index), cbar=False);


# # 10.  Что делать, если темы плохо выделились? 
# 
# * Больше предобработки данных, надо выкинуть больше стоп-слов, чтобы из-за них не возникали мусорные тематики 
# * Больше итераций обучения модели 
# * Попробовать выделять другое количество тематик, не факт что выбрали самое клёвое $T$ из всех возможных! 
# * Можно попробовать поперебирать $T$ и позамерять такую метрику качества как перплексия, но об этом почитаете в интернете. 
# * Можно собрать ещё данных. 

# # Полезные материалы
# 
# * [Тетрадка с семинара по API вконтакте](https://nbviewer.jupyter.org/github/FUlyankin/HSE_Data_Culture/blob/master/ML_for_marketing_2019/sems/sem10_vk/vk_parser_full.ipynb) 
# * [Расширенная тетрадка по мотивам семинара,](https://nbviewer.jupyter.org/github/FUlyankin/HSE_Data_Culture/blob/master/ML_for_marketing_2019/sems/sem10_vk/vk_parser_download.ipynb)  которой я скачивал мемы 
# * [Туториал по тому как качать вконтакте,](https://nbviewer.jupyter.org/github/FUlyankin/ekanam_grand_research/blob/master/0.%20vk_parser_tutorial.ipynb) в нём есть функция, которая ускоряет скачку в 25 раз. Она обычно довольно полезна для скачки лайков и комментариев. 
# 
# * [Небольшой проект от меня и моего Димы](https://github.com/DmitrySerg/top-russian-music) про рэпчик и тематическое моделирование 
# * [Дима рассказывает про этот проект на PyData](https://www.youtube.com/watch?v=MEBjnGaHsmw)  (видос полчаса без регистрации и смс)

#  
