from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
import operator


def create_cloud(file_name, text, max_font_size=200, max_words=100, width=1200, height=800):
    try:
        wc = WordCloud(max_font_size=max_font_size, max_words=max_words,
                       width=width, height=height, background_color="white")
        wc.generate(text)
        wc.to_file(file_name)
        return True
    except Exception as e:
        return str(e)


def sort(data):
    popular = {}
    content = ''

    for category in data:
        words = category[1]
        for i in range(len(words)):
            if i > 20:
                break
            popular.setdefault(words[i][0], words[i][1])
    sort_popular = sorted(popular.items(), key=operator.itemgetter(1), reverse=True)
    for item in sort_popular:
        content += ' ' + item[0]
    return content


if __name__ == '__main__':
    files = ['all', 'worldofwarcraft', 'leagueoflegends', 'fortnite', 'wgcsgo', 'dota2']

    for file in files:
        data = np.load('data/{}_top_words_text.npy'.format(file), allow_pickle=True)
        content = sort(data)
        status = create_cloud('data/text_{}.jpg'.format(file), content)
        print(status)