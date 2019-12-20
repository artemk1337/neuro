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









