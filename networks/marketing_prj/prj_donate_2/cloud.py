from wordcloud import WordCloud
import numpy as np
import operator
from PIL import Image


data = np.load('all_words_repost.npy', allow_pickle=True)
print(data)
popular = {}
for category in data:
    print(category)
    words = category[1]
    for i in range(len(words)):
        if i > 20:
            break
        popular.setdefault(words[i][0], words[i][1])
sorted_words = sorted(popular.items(), key=operator.itemgetter(1), reverse=True)
data = ''
for item in sorted_words:
    data += ' ' + item[0]


wc = WordCloud(max_font_size=200, max_words=150, width=1500, height=1500, background_color="white")
wc.generate(data)
wc.to_file("test.png")
a = wc.to_array()
a = Image.fromarray(a)
a.show()
