from bs4 import BeautifulSoup
import requests
import re
import numpy as np
import json


def parse(url, matches_links):
    page = requests.get(url)
    # Success - 200
    if page.status_code != 200:
        exit(-1)
    soup = BeautifulSoup(page.text, "html.parser")
    # print(soup)
    new = []
    for i in soup.findAll('a', class_='a-reset'):
        new.append(i.get('href'))
    for i in new:
        if re.search(r'\bmatches\b', i):
            matches_links.append(i)
    del new


def choose_links():
    matches_links = []
    # Первая страница
    parse('https://www.hltv.org/results', matches_links)
    # Остальные страницы
    for i in range(1, 30):  # max - 4930
        # print(i)
        parse(f'https://www.hltv.org/results?offset={i * 100}', matches_links)
    # Сохранение
    print(f'Матчей - {len(matches_links)}')
    with open('matches_links.json', 'w') as file:
        json.dump(matches_links, file)
    return matches_links


def get_info(m_links):
    def get_rank():
        rank = []
        for i in range(1, 3):
            tmp = BeautifulSoup(page.text, "html.parser").findAll('div', class_=f'team{i}-gradient')[0].findAll('a')[0]\
                .get('href')
            rank_1 = requests.get(f'https://www.hltv.org{tmp}')
            if rank_1.status_code != 200:
                exit(-1)
            try:
                rank_1 = BeautifulSoup(rank_1.text, "html.parser").findAll('div', class_='profile-team-stat')[0].findAll('span')[0].text
                rank.append(rank_1.split('#')[1])
            except:
                rank.append(201)
        return rank[0], rank[1]

    def get_team_name():
        return BeautifulSoup(page.text, "html.parser").findAll('div', class_='teamName')[0].text, \
               BeautifulSoup(page.text, "html.parser").findAll('div', class_='teamName')[1].text

    def get_date():
        return int(int(BeautifulSoup(page.text, "html.parser").findAll('div', class_='date')[0].get('data-unix')) / 1000)

    def get_score():
        return BeautifulSoup(page.text, "html.parser").findAll('div', class_='team1-gradient')[0].findAll('div')[1]\
        .text, BeautifulSoup(page.text, "html.parser").findAll('div', class_='team2-gradient')[0].findAll('div')[1].text

    def get_maps_score():
        maps = []
        maps_score = []
        for i in BeautifulSoup(page.text, "html.parser").findAll('div', class_='results'):
            maps_score.append(i.text)
        for i in range(len(maps_score)):
            maps.append(BeautifulSoup(page.text, "html.parser").findAll('div', class_='mapname')[i].text)
        new_maps = []
        for i in range(len(maps_score)):
            new_maps.append([maps[i], maps_score[i].split()[0].split(':')[0], maps_score[i].split()[0].split(':')[1]])
        del maps, maps_score
        return new_maps

    def get_players(s):
        players = []
        page = requests.get(f'https://www.hltv.org{s}')
        for i in BeautifulSoup(page.text, "html.parser").findAll('td', class_='player'):
            for k in i.findAll('a'):
                players.append(k.get('href'))
        players = players[int(len(players) / 4):int((len(players) / 4) * 3)]
        names = []
        stat = []
        for i in players:
            names.append(i.split('/')[3])
            page = requests.get(f'https://www.hltv.org{i}')
            stat.append(BeautifulSoup(page.text, "html.parser").findAll('span', class_='statsVal')[0].text)
        for i in range(len(players)):
            with open('players.json', 'a+') as file_1:
                json.dump({f'{players[i]}': {'rate': stat[i]}}, file_1, indent=4, separators=(',', ': '))

    def get_all_and_save(s):
        try:
            table = {}
            date = get_date()
            team_1, team_2 = get_team_name()
            rank_1, rank_2 = get_rank()
            score_1, score_2 = get_score()
            maps = get_maps_score()
            table[f'https://www.hltv.org{s}'] = {'date': date,
                                                 'rank_1': rank_1,
                                                 'rank_2': rank_2,
                                                 'team_1': team_1,
                                                 'team_2': team_2,
                                                 'score_1': score_1,
                                                 'score_2': score_2,
                                                 'maps': maps}
            with open('table.json', 'a+') as file:
                json.dump(table, file, indent=4, separators=(',', ': '))
        except:
            pass
        try:
            get_players(s)
        except:
            pass

    for i in range(len(m_links)):
        print(i)
        page = requests.get(f'https://www.hltv.org{m_links[i]}')
        if page.status_code != 200:
            exit(-1)
        get_all_and_save(m_links[i])


def stats_train(m_links):
    def get_players(s):
        players = []
        page = requests.get(s)
        for i in BeautifulSoup(page.text, "html.parser").findAll('td', class_='player'):
            for k in i.findAll('a'):
                players.append(k.get('href'))
        players = players[int(len(players) / 4):int((len(players) / 4) * 3)]
        names = []
        stat = []
        for i in players:
            names.append(i.split('/')[3])
            page = requests.get(f'https://www.hltv.org{i}')
            stat.append(BeautifulSoup(page.text, "html.parser").findAll('span', class_='statsVal')[0].text)
        return stat

    def train_data(s, number, table):
        players_stat = get_players(s)
        page = requests.get(s)
        if page.status_code != 200:
            exit(-1)
        tmp1 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='team1-gradient')[0].findAll('div')[1].text
        tmp2 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='team2-gradient')[0].findAll('div')[1].text
        g_score = int(tmp1) / (int(tmp1) + int(tmp2))

        tmp1 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='team1-gradient')[0].findAll('a')[0].get(
            'href')
        tmp2 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='team2-gradient')[0].findAll('a')[0].get(
            'href')
        link1 = f'https://www.hltv.org{tmp1}'
        link2 = f'https://www.hltv.org{tmp2}'
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
            arr = (BeautifulSoup(page.text, "html.parser").findAll('div', class_='past-matches')[0].findAll('tr', class_='table')[i].text.split())
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
        table[f'{number}'] = {'res': g_score,
                                'rank_1': rank[0][0],
                                'rank_2': rank[1][0],
                                'score_1': [{f'{i}': scores[0][i] for i in range(len(scores[0]))}],
                                'score_2': [{f'{i}': scores[1][i] for i in range(len(scores[1]))}],
                                'ranks_1': [{f'{i}': rank1[0][i] for i in range(len(rank1[0]))}],
                                'ranks_2': [{f'{i}': rank1[1][i] for i in range(len(rank1[1]))}],
                                'player_stat_1': [{f'{i}': players_stat[i] for i in range(5)}],
                                'player_stat_2': [{f'{i - 5}': players_stat[i] for i in range(5, 10)}],
                                }
        with open('match_train.json', 'w') as file:
            json.dump(table, file, indent=4, separators=(',', ': '))

    table = {}
    for i in range(len(m_links)):
        print(f'Матч №{i}')
        page = requests.get(f'https://www.hltv.org{m_links[i]}')
        if page.status_code != 200:
            exit(-1)
        try:
            train_data(f'https://www.hltv.org{m_links[i]}', i, table)
        except:
            pass


m_links = choose_links()
# get_info(m_links)
stats_train(m_links)

