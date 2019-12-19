from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
import json
import re
import csv


public_name = ['wgcsgo', 'leagueoflegends', 'fortnite', 'dota2', 'worldofwarcraft']
hot_words = ['розыгр', 'выигр', 'получ', 'конкурс', 'разыгр', 'приз', 'услов', 'участ']
hot_words_del = ['https', 'vk', 'com', 'http', 'ru',
                 'https_vk', 'youtube', 'www', 'club', 'id']


def convert_text():
    for fn in public_name:
        text = []
        repost = []
        with open(f'data/{fn}.json', 'r') as f:
            data = json.load(f)
        text_k = 0
        repost_k = 0
        ad_post = 0
        for i in data.keys():
            # print(data[i])
            for k in data[i]['wall']['items']:
                text.append(k['text'])
                text_k += 1
                try:
                    repost.append(k['copy_history'][0]['text'])
                    repost_k += 1
                    for word in hot_words:
                        if re.search(word.lower(), k['copy_history'][0]['text'].lower()) or\
                           re.search(word.lower(), k['text'].lower()):
                            ad_post += 1
                            break
                except:
                    pass
        # print(repost)
        print(f'{round(repost_k / text_k * 100)}% - {fn} - Репосты')
        print(f'{round(ad_post / repost_k * 100)}% из них розыгрыши')
        fig1, ax1 = plt.subplots()
        plt.title(f'{fn}')
        ax1.pie([text_k - repost_k, repost_k - ad_post, ad_post], labels=['Прочее', 'Репосты', 'Розыгрыши'], autopct='%1.2f%%')
        plt.savefig(f'data/{fn}/{fn}_type_post.png')
        # plt.show()
        text_new = []
        for i in text:
            if i != '':
                text_new.append(i)
        # with open(f'data/{fn}/{fn}_all_text.txt', 'w', encoding='utf-8') as f:
            # f.write(text_new)
        np.save(f'data/{fn}/{fn}_all_text', np.asarray(text_new))
        repost_new = []
        for i in repost:
            if i != '':
                repost_new.append(i)
        np.save(f'data/{fn}/{fn}_all_repost', np.asarray(repost_new))
        # with open(f'data/{fn}/{fn}_all_repost.txt', 'w', encoding='utf-8') as f:
            # f.write(repost_new)


"""convert text to 2D-array and save as numpy"""


# convert_text()


import gensim

# Scikit learn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Keras
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras import utils
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import pickle
import os
import time
from tqdm import tqdm

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

import pyLDAvis
import nltk
from nltk.corpus import stopwords
import pyLDAvis.gensim

import pymorphy2
# from pymystem3 import Mystem


def lda(Filename, tt):
    if tt == 1:
        data = np.load(f'data/{Filename}/{Filename}_all_repost.npy')
    else:
        arr = 0
        for i in public_name:
            data = np.load(f'data/{i}/{i}_all_repost.npy')
            if arr == 0:
                arr = data
            else:
                arr = np.concatenate((arr, data))
        data = arr
        np.save('data/all_repost')

    nltk.download('stopwords')
    nltk.download('wordnet')
    stopwords_ru = stopwords.words('russian')
    stopwords_en = stopwords.words('english')

    x_train = [gensim.utils.simple_preprocess(text) for text in data]

    # Количество слов
    x_train = [x for x in x_train if len(x) > 50]

    # Работает ОЧЕНЬ МЕДЛЕННО!
    # lemm = Mystem()
    # Работает шикарно!
    morph = pymorphy2.MorphAnalyzer()

    # Начальная форму
    x_train = [[morph.parse(word)[0].normal_form for word in i] for i in x_train]

    # Удаляю слова
    x_train = [[word for word in x if word not in stopwords_ru] for x in x_train]
    x_train = [[word for word in x if word not in stopwords_en] for x in x_train]

    fin = []
    for i in x_train:
        arr = []
        for k in i:
            c = 0
            for t in hot_words_del:
                if re.search(t, k):
                    c += 1
            if c == 0:
                arr.append(k)
        fin.append(arr)

    x_train = fin

    """Join compound words (Example: cs_go or more)"""

    """
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(x_train, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[x_train], threshold=100)
    
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
    """

    # Form Bigrams
    # texts = make_bigrams(x_train)
    texts = x_train

    """<=============CREATE DICT=============>"""

    id2word = corpora.Dictionary(texts)
    corpus = [id2word.doc2bow(text) for text in texts]

    # слово должно встретиться хотябы 5 раз и не более чем в 50% документов
    id2word.filter_extremes(no_below=5, no_above=0.5)
    corpus = [id2word.doc2bow(text) for text in texts]

    from collections import defaultdict
    import itertools

    def word_freq_plot(dictionary, corpus, k2=100, k1=0):
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

        plt.figure(figsize=(22, 10))
        plt.bar(indices, frequency)
        plt.xticks(indices, word, rotation='vertical', fontsize=12)
        plt.tight_layout()
        if tt == 1:
            if not os.path.isdir(f'data/{Filename}/LDA'):
                os.mkdir(f'data/{Filename}/LDA')
            plt.savefig(f'data/{Filename}/LDA/most_popular_words.jpg')
        else:
            plt.savefig(f'data/most_popular_words.jpg')
        plt.show()

    word_freq_plot(id2word, corpus)

    """Cut too popular words"""

    '''
    print(len(id2word))
    arr = np.zeros((len(id2word)))
    print(arr.shape)
    for i in corpus:
        for k in i:
            print(int(k[0]))
            arr[k[0]] += 1
    
    
    plt.plot(arr)
    plt.show()
    
    bad_id = np.array([x for x in arr if x > 100])
    id2word.filter_tokens(bad_ids=bad_id)
    corpus = [id2word.doc2bow(text) for text in texts]
    '''

    print(f'Posts - {len(texts)}')

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=5)

    if tt == 1:
        if not os.path.isdir(f'data/{Filename}/LDA'):
            os.mkdir(f'data/{Filename}/LDA')
        lda_model.save(f'data/{Filename}/LDA/LDA_model')
        # pyLDAvis.enable_notebook()  # Only in notebook
        visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(visualisation, f'data/{Filename}/LDA/LDA_Visualization_{Filename}.html')
    else:
        lda_model.save(f'LDA_model')
        # pyLDAvis.enable_notebook()  # Only in notebook
        visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(visualisation, f'LDA_Visualization_.html')
    print(lda_model.print_topics())

    topics = lda_model.show_topics(num_topics=5, num_words=50, formatted=False)
    print(topics)

    words = []
    for i in range(len(topics)):
        tmp = []
        for k in topics[i][1]:
            tmp.append(k[0])
        words.append(tmp)


for i in public_name:
    # lda(i, 1)
    pass

lda('aaa', 0)





