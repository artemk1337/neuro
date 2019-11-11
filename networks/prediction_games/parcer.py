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
    for i in range(1, 2):  # max - 4930
        # print(i)
        parse(f'https://www.hltv.org/results?offset={i * 100}', matches_links)
    # Сохранение
    with open('matches_links.json', 'w') as file:
        json.dump(matches_links, file)
    return matches_links


def get_info(m_links, table):
    def get_rank():
        rank = []
        for i in range(1, 3):
            tmp = BeautifulSoup(page.text, "html.parser").findAll('div', class_=f'team{i}-gradient')[0].findAll('a')[0]\
                .get('href')
            rank_1 = requests.get(f'https://www.hltv.org{tmp}')
            if rank_1.status_code != 200:
                exit(-1)
            rank_1 = BeautifulSoup(rank_1.text, "html.parser").findAll('div', class_='profile-team-stat')[0].findAll('span')[0].text
            if rank == 0:
                rank.append(0)
            else:
                rank.append(rank_1.split('#')[1])
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

    def get_all_and_save():
        date = get_date()
        team_1, team_2 = get_team_name()
        rank_1, rank_2 = get_rank()
        score_1, score_2 = get_score()
        maps = get_maps_score()

        table[f'{m_links[0]}'] = {'date': date,
                                  'rank_1': rank_1,
                                  'rank_2': rank_2,
                                  'team_1': team_1,
                                  'team_2': team_2,
                                  'score_1': score_1,
                                  'score_2': score_2,
                                  'maps': maps}
        with open('table.json', 'w') as file:
            json.dump(table, file, indent=4, separators=(',', ': '))

    for i in range(len(m_links)):
        print(i)
        page = requests.get(f'https://www.hltv.org{m_links[i]}')
        if page.status_code != 200:
            exit(-1)
        get_all_and_save()


table = {}
m_links = choose_links()
get_info(m_links, table)
