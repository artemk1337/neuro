from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
import json
import re


public_name = ['wgcsgo', 'leagueoflegends', 'fortnite', 'dota2', 'worldofwarcraft']
hot_words = ['розыгр', 'выигр', 'получ', 'конкурс', 'разыгр', 'приз', 'услов', 'участ']


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

# Utility
import numpy as np
import pickle
import tempfile
import os
import time
import logging
import multiprocessing
from tqdm import tqdm





import re
import numpy as np
import pandas as pd
from pprint import pprint
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# spacy for lemmatization
import spacy
# Plotting tools
import pyLDAvis
import nltk
from nltk.corpus import stopwords
import pyLDAvis.gensim
from pymystem3 import Mystem


lemm = Mystem()


data = np.load('data/wgcsgo/wgcsgo_all_repost.npy')

nltk.download('stopwords')
nltk.download('wordnet')
stopwords_ru = stopwords.words('russian')
stopwords_en = stopwords.words('english')


x_train = [gensim.utils.simple_preprocess(text) for text in data]
x_train = [x for x in x_train if len(x) > 100]
x_train = [[word for word in x if word not in stopwords_ru] for x in x_train]
x_train = [[word for word in x if word not in stopwords_en] for x in x_train]
print(x_train[0])
x_train = [[lemm.lemmatize(word)[0] for word in i] for i in x_train[:2]]
print(x_train[0])
quit()

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


# Form Bigrams
texts = make_bigrams(x_train)

# Create Dictionary
id2word = corpora.Dictionary(texts)
# Create Corpus
corpus = [id2word.doc2bow(text) for text in texts]

# слово должно встретиться хотябы 5 раз и не более чем в 20% документов
id2word.filter_extremes(no_below=5, no_above=0.2)
corpus = [id2word.doc2bow(text) for text in texts]
print(len(id2word))
print([id2word[i] for i in range(len(id2word))])


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

'''lda_model = models.ldamulticore.LdaMulticore(corpus=corpus,
                                              id2word=id2word,
                                              num_topics=10,
                                              random_state=42,
                                              passes=20)'''


print(lda_model.print_topics())


'''coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)'''

# Visualize the topics
# pyLDAvis.enable_notebook()  # Only in notebook
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')



