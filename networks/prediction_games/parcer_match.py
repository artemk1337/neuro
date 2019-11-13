from bs4 import BeautifulSoup
import requests
import numpy as np
import json
import keras
import _pickle as cPickle


def get_players(s1, s2):
    players = []
    page = requests.get(s1)
    for i in BeautifulSoup(page.text, "html.parser").findAll('div', class_='bodyshot-team g-grid')[0].findAll('a'):
        players.append(i.get('href'))
    names = []
    stat = []
    for i in players:
        names.append(i.split('/')[3])
        page = requests.get(f'https://www.hltv.org{i}')
        stat.append(BeautifulSoup(page.text, "html.parser").findAll('span', class_='statsVal')[0].text)
    players = []
    page = requests.get(s2)
    for i in BeautifulSoup(page.text, "html.parser").findAll('div', class_='bodyshot-team g-grid')[0].findAll('a'):
        players.append(i.get('href'))
    names = []
    for i in players:
        names.append(i.split('/')[3])
        page = requests.get(f'https://www.hltv.org{i}')
        stat.append(BeautifulSoup(page.text, "html.parser").findAll('span', class_='statsVal')[0].text)
    return stat


def train_data(s, table):
    page = requests.get(s)
    if page.status_code != 200:
        exit(-1)
    tmp1 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='team1-gradient')[0].findAll('a')[0].get(
        'href')
    tmp2 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='team2-gradient')[0].findAll('a')[0].get(
        'href')
    link1 = f'https://www.hltv.org{tmp1}'
    link2 = f'https://www.hltv.org{tmp2}'
    players_stat = get_players(link1, link2)

    rank = [[], []]
    try:
        rank_1 = BeautifulSoup(requests.get(link1).text, "html.parser").findAll(
            'div', class_='profile-team-stat')[0].findAll('span')[
            0].text
        rank[0].append(rank_1.split('#')[1])
    except:
        rank[0].append(1000)
    try:
        rank_2 = BeautifulSoup(requests.get(link2).text, "html.parser").findAll(
            'div', class_='profile-team-stat')[0].findAll('span')[
            0].text
        rank[1].append(rank_2.split('#')[1])
    except:
        rank[1].append(1000)

    scores = [[], []]
    for i in range(10):
        arr = (BeautifulSoup(page.text, "html.parser").findAll('div', class_='past-matches')[0].findAll('tr', class_='table')
                   [i].text.split())
        res1 = int(arr[-3])
        res2 = int(arr[-1])
        if res2 == 0:
            scores[int(i / 5)].append(1)
        elif res1 == 0:
            scores[int(i / 5)].append(0)
        else:
            scores[int(i / 5)].append(res1 / (res1 + res2))
    rank1 = [[], []]
    for i in range(10):
        tmp = BeautifulSoup(page.text, "html.parser").findAll('td', class_='opponent')[i].findAll('a')[0].get(
            'href')
        try:
            rank_1 = BeautifulSoup(requests.get(f'https://www.hltv.org{tmp}').text, "html.parser").findAll(
                'div', class_='profile-team-stat')[0].findAll('span')[0].text
            rank1[int(i / 5)].append(int(rank_1.split('#')[1]))
        except:
            rank1[int(i / 5)].append(1000)
    for i in range(len(rank)):
        rank[i] = [int(rank[i][k]) for k in range(len(rank[i]))]
    for i in range(len(rank1)):
        rank1[i] = [int(rank1[i][k]) for k in range(len(rank1[i]))]
    for i in range(len(scores)):
        scores[i] = [float(scores[i][k]) for k in range(len(scores[i]))]
    players_stat = [float(i) for i in players_stat]
    table = {'rank_1': rank[0][0],
             'rank_2': rank[1][0],
             'score_1': [{f'{i}': scores[0][i] for i in range(len(scores[0]))}],
             'score_2': [{f'{i}': scores[1][i] for i in range(len(scores[1]))}],
             'ranks_1': [{f'{i}': rank1[0][i] for i in range(len(rank1[0]))}],
             'ranks_2': [{f'{i}': rank1[1][i] for i in range(len(rank1[1]))}],
             'player_stat_1': [{f'{i}': players_stat[i] for i in range(5)}],
             'player_stat_2': [{f'{i - 5}': players_stat[i] for i in range(5, 10)}],
             }
    return table


def parse(s):
    table = {}
    page = requests.get(f'{s}')
    if page.status_code != 200:
        exit(-1)
    return train_data(f'{s}', table)


def transform(a):
    table = []
    # print(a['0'])
    tmp = []
    tmp.append(a['rank_1'])
    tmp.append(a['rank_2'])
    for k in a['score_1']:
        for t in k:
            tmp.append(k[t])
    for k in a['ranks_1']:
        for t in k:
            tmp.append(k[t])
    for k in a['score_2']:
        for t in k:
            tmp.append(k[t])
    for k in a['ranks_2']:
        for t in k:
            tmp.append(k[t])
    for k in a['player_stat_1']:
        for t in k:
            tmp.append(k[t])
    for k in a['player_stat_2']:
        for t in k:
            tmp.append(k[t])
    table.append(tmp)
    del tmp
    table = np.asarray(table).astype(np.float)
    return table


def main(s):
    a = parse(s)
    a = transform(a)
    m = keras.models.load_model('model')
    clf = cPickle.load(open('rf.pkl', 'rb'))
    clf1 = cPickle.load(open('rf1.pkl', 'rb'))
    clf2 = cPickle.load(open('rf2.pkl', 'rb'))
    clf3 = cPickle.load(open('rf3.pkl', 'rb'))
    print([f'{int(clf.predict(a[:])[0] * 100)}%',
           f'{int(clf1.predict(a[:])[0] * 100)}%',
           f'{int(clf2.predict(a[:])[0] * 100)}%',
           f'{int(clf3.predict(a[:])[0] * 100)}%',
           f'{int(m.predict(a[:])[0, 0] * 100)}%'])
    res = (clf.predict(a[:])[0] +
              clf1.predict(a[:])[0] +
              clf2.predict(a[:])[0] +
              clf3.predict(a[:])[0] +
              m.predict(a[:])[0, 0]) / 5
    print(res)


main('https://www.hltv.org/matches/2337715/forze-vs-giants-united-masters-league-season-2')

