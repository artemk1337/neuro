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
        print(i)
        parse(f'https://www.hltv.org/results?offset={i * 100}', matches_links)
    # Сохранение
    with open('matches_links.json', 'w') as file:
        json.dump(matches_links, file)
    return matches_links


def get_info(m_links, table):
    page = requests.get(f'https://www.hltv.org{m_links[0]}')
    if page.status_code != 200:
        exit(-1)
    date = int(int(BeautifulSoup(page.text, "html.parser").findAll('div', class_='date')[0].get('data-unix')) / 1000)
    team_1 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='teamName')[0].text
    team_2 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='teamName')[1].text
    try:
        try:
            score_1 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='team1-gradient')[0].findAll('div', class_='lost')[0].text
            score_2 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='team2-gradient')[0].findAll('div', class_='won')[0].text
        except:
            score_1 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='team1-gradient')[0].findAll('div',class_='won')[0].text
            score_2 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='team2-gradient')[0].findAll('div',class_='lost')[0].text
    except:
        score_1 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='team1-gradient')[0].findAll('div', class_='none')[0].text
        score_2 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='team2-gradient')[0].findAll('div', class_='none')[0].text
    print(score_1, score_2)
    table[f'{m_links[0]}'] = {'date': date}
    with open('table.json', 'w') as file:
        json.dump(table, file, indent=4, separators=(',', ': '))






table = {}
m_links = choose_links()
get_info(m_links, table)
