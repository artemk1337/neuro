from bs4 import BeautifulSoup
import requests
import re
import numpy as np
import json
import csv
from threading import Thread

counter_last_matches = 10


def parse(url):
    matches_links = []
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
    return matches_links


def parse_links():
    # Первая страница
    matches_links = parse('https://www.hltv.org/results')
    with open('data/matches_links_0.json', 'w') as file:
        json.dump(matches_links, file)
    # Остальные страницы
    for i in range(1, 5):  # max - 4930
        # print(i)
        matches_links = parse(f'https://www.hltv.org/results?offset={i * 100}')
        with open(f'data/matches_links_{i}.json', 'w') as file:
            json.dump(matches_links, file)


# Optimized
# Get players from team
# BAD!!!
def get_players(url):
    page = requests.get(url)
    tmp1 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='team1-gradient')[0].findAll('a')[0].get(
        'href')
    tmp2 = BeautifulSoup(page.text, "html.parser").findAll('div', class_='team2-gradient')[0].findAll('a')[0].get(
        'href')
    s1 = f'https://www.hltv.org{tmp1}'
    s2 = f'https://www.hltv.org{tmp2}'

    kd = []
    kpr = []
    hd = []
    mp = []
    dpr = []
    rc = []
    names = []

    players = []
    page = requests.get(s1)
    page = BeautifulSoup(page.text, "html.parser")
    for i in page.findAll('div', class_='bodyshot-team g-grid')[0].findAll('a'):
        players.append(i.get('href'))
    for i in players:
        names.append(i.split('/')[3])
        page = requests.get(f'https://www.hltv.org{i}')
        page = BeautifulSoup(page.text, "html.parser")
        kd.append(page.findAll('span', class_='statsVal')[0].text)
        kpr.append(page.findAll('span', class_='statsVal')[1].text)
        hd.append(page.findAll('span', class_='statsVal')[2].text.split('%')[0])
        mp.append(page.findAll('span', class_='statsVal')[3].text)
        dpr.append(page.findAll('span', class_='statsVal')[4].text)
        rc.append(page.findAll('span', class_='statsVal')[5].text.split('%')[0])

    players = []
    page = requests.get(s2)
    page = BeautifulSoup(page.text, "html.parser")
    for i in page.findAll('div', class_='bodyshot-team g-grid')[0].findAll('a'):
        players.append(i.get('href'))
    for i in players:
        names.append(i.split('/')[3])
        page = requests.get(f'https://www.hltv.org{i}')
        page = BeautifulSoup(page.text, "html.parser")
        kd.append(page.findAll('span', class_='statsVal')[0].text)
        kpr.append(page.findAll('span', class_='statsVal')[1].text)
        hd.append(page.findAll('span', class_='statsVal')[2].text.split('%')[0])
        mp.append(page.findAll('span', class_='statsVal')[3].text)
        dpr.append(page.findAll('span', class_='statsVal')[4].text)
        rc.append(page.findAll('span', class_='statsVal')[5].text.split('%')[0])
    return kd, kpr, hd, mp, dpr, rc, names


def check_size(arr, i):
    if len(arr) != i:
        print('ERROR!!!')
        exit(-1)


# Optimized
# Get players from match
def get_players_1(url, tick):
    players = []

    page = requests.get(url)
    page = BeautifulSoup(page.text, "html.parser")
    page = page.findAll('td', class_='player')
    for i in page:
        for k in i.findAll('a'):
            players.append(k.get('href'))

    if tick == 0:
        players = players[:5]
    elif tick == -1:
        players = players[5:15]
    else:
        players = players[-5:]
    # print(players)
    names = []
    kd = []
    kpr = []
    hd = []
    mp = []
    dpr = []
    rc = []

    for i in players:
        names.append(i.split('/')[3])
        page = requests.get(f'https://www.hltv.org{i}')
        page = BeautifulSoup(page.text, "html.parser")
        kd.append(page.findAll('span', class_='statsVal')[0].text)
        kpr.append(page.findAll('span', class_='statsVal')[1].text)
        hd.append(page.findAll('span', class_='statsVal')[2].text.split('%')[0])
        mp.append(page.findAll('span', class_='statsVal')[3].text)
        dpr.append(page.findAll('span', class_='statsVal')[4].text)
        rc.append(page.findAll('span', class_='statsVal')[5].text.split('%')[0])
    new = kd + kpr + hd + mp + dpr + rc
    for i in range(len(new)):
        new[i] = float(new[i])
    if tick == -1:
        check_size(new, 60)
    else:
        check_size(new, 30)
    return new


def get_data(url):
    # Optimized
    def get_date(url):
        page = requests.get(url)
        return int(int(BeautifulSoup(page.text, "html.parser").findAll('div', class_='date')[0] \
                       .get('data-unix')) / 1000)

    # Optimized
    def get_team_name(url):
        page = requests.get(url)
        page = BeautifulSoup(page.text, "html.parser")
        tmp1 = page.findAll('div', class_='teamName')[0].text
        tmp2 = page.findAll('div', class_='teamName')[1].text
        check_size([tmp1, tmp2], 2)
        return [tmp1, tmp2]

    # Optimized
    def get_g_score(url):
        page = requests.get(url)
        tmp = BeautifulSoup(page.text, "html.parser")
        tmp1 = tmp.findAll('div', class_='team1-gradient')[0] \
            .findAll('div')[1].text
        tmp2 = tmp.findAll('div', class_='team2-gradient')[0] \
            .findAll('div')[1].text
        return float(int(tmp1) / (int(tmp1) + int(tmp2)))

    # Optimized
    def get_history_score(url, current_date):
        page = requests.get(url)
        page = BeautifulSoup(page.text, "html.parser")
        tmp1 = int(page \
                   .find_all('div', class_='flexbox-column flexbox-center grow right-border')[0] \
                   .find_all('div', class_='bold')[0] \
                   .text)
        # print(tmp1)
        tmp2 = int(page \
                   .find_all('div', class_='flexbox-column flexbox-center grow left-border')[0] \
                   .find_all('div', class_='bold')[0] \
                   .text)
        # print(tmp2)
        tmp3 = page \
            .find_all('tr', attrs=re.compile(".*row nowrap.*"))
        counter = 0
        score = 0
        for i in range(len(tmp3)):
            tmp3_date = int(int(tmp3[i].find_all('span')[0].get('data-unix')) / 1000)
            if 0 < current_date - tmp3_date < 15552000:  # Last 6 months
                counter += 1
                # print(current_date - tmp3_date)
                tmp3_ = tmp3[i].find_all('td', 'result')[0].text
                tmp3_ = tmp3_.split()
                score += int(tmp3_[0]) - int(tmp3_[2])
                # print(tmp3_)
            # print(tmp3_date)
        if counter != 0:
            score = ((score / counter) + 16) / 32
        else:
            score = 0.5
        check_size([tmp1, tmp2, score], 3)
        return [tmp1, tmp2, score]

    # Optimized
    def get_prize_pull(url):
        page = requests.get(url)
        url1 = BeautifulSoup(page.text, "html.parser") \
            .findAll('div', class_='event text-ellipsis')[0] \
            .find_all('a')[0] \
            .get('href')
        page = requests.get(f'https://www.hltv.org{url1}')
        txt = BeautifulSoup(page.text, "html.parser") \
            .findAll('td', class_='prizepool text-ellipsis')[0].get('title')
        try:
            txt = txt.split()
            txt = txt[0].split('$')[1]
            txt = txt.split(',')
            res = ''
            for i in range(len(txt)):
                res += txt[i]
            res = int(res)
            return res
        except:
            return 0

    # Optimized
    def all_stat_team(url, curr_date):
        # Optimized
        def get_win_streak_and_rate(s):
            tmp = s.find_all('div', 'highlighted-stat')
            try:
                tmp1 = float(tmp[0].find_all('div', 'stat')[0].text)
            except:
                tmp1 = 0
            try:
                tmp2 = float(tmp[1].find_all('div', 'stat')[0].text.split('%')[0])
            except:
                tmp2 = 0
            check_size([tmp1, tmp2], 2)
            return [tmp1, tmp2]

        # Optimized
        def get_rank(s):
            try:
                rank_1 = s[0].findAll('span')[0].text
                return int(rank_1.split('#')[1])
            except:
                return 1000

        # Optimized
        def get_avarage_age(s):
            try:
                return float(s[2].findAll('span')[0].text)
            except:
                return 0

        # Optimized
        def get_weeks_in_top30_for_core(s):
            try:
                return float(s[1].findAll('span')[0].text)
            except:
                return 0

        # Optimized
        def world_ranking_avarage_age_weeks(s):
            tmp = BeautifulSoup(requests.get(s).text, "html.parser")
            arr = tmp.findAll('div', class_='profile-team-stat')
            tmp = [get_rank(arr), get_weeks_in_top30_for_core(arr),
                   get_avarage_age(arr), *get_win_streak_and_rate(tmp)]
            check_size(tmp, 5)
            return tmp

        def last_20_matches(s):
            team_name = s.split('/')[-1]
            print(f'parse {counter_last_matches} last matches for {team_name}')
            page = requests.get(s)
            page = BeautifulSoup(page.text, "html.parser").find_all('a', 'moreButton')[1].get('href')
            page = requests.get(f'https://www.hltv.org{page}')
            page = BeautifulSoup(page.text, "html.parser").find_all('div', 'result-con')
            # Собрали список последних матчей команды
            i = 0
            g_i = 0
            res = []
            score = []
            prize = []
            players = []
            history = []

            while g_i < counter_last_matches:
                tmp = page[i]
                tmp2 = tmp.find_all('a', 'a-reset')[0].get('href')
                match_link = f'https://www.hltv.org{tmp2}'
                # Проверяем, что матч был раньше
                date = get_date(match_link)
                if 0 < curr_date - date:
                    try:
                        tmp1 = tmp.find_all('td', 'result-score')[0].text.split()
                        # Счет матча
                        score_tmp = (int(tmp1[0]) / (int(tmp1[0]) + int(tmp1[2])))
                        # Приз турнира
                        prize_tmp = get_prize_pull(f'https://www.hltv.org{tmp2}')
                        # Save match link
                        # История встреч
                        history_tmp = get_history_score(match_link, date)
                        # Переходим на страницу матча
                        tmp2 = requests.get(f'https://www.hltv.org{tmp2}')
                        tmp2 = BeautifulSoup(tmp2.text, "html.parser")
                        tmp2 = tmp2.find_all('div', re.compile("team.*-gradient"))
                        # Собираем инфу на команду соперника
                        # Проверяем, что команда не та же самая
                        if str(tmp2[0].find_all('a')[0].get('href').split('/')[-1]) != str(team_name):
                            players_tmp = get_players_1(match_link, 0)
                            tmp2 = tmp2[0].find_all('a')[0].get('href')
                            res.append(world_ranking_avarage_age_weeks(f'https://www.hltv.org{tmp2}'))
                        else:
                            players_tmp = get_players_1(match_link, 1)
                            tmp2 = tmp2[1].find_all('a')[0].get('href')
                            res.append(world_ranking_avarage_age_weeks(f'https://www.hltv.org{tmp2}'))
                        score.append(score_tmp)
                        prize.append(prize_tmp)
                        players.append(players_tmp)
                        history.append(history_tmp)
                        # print(f'https://www.hltv.org{tmp2}')
                        print(f'parse match {g_i}')
                        g_i += 1
                    except:
                        pass
                i += 1
            return [*res, score, prize, *players, *history]

        page = requests.get(url)
        page = BeautifulSoup(page.text, "html.parser")
        tmp1 = page.findAll('div', class_='team1-gradient')[0].findAll('a')[0].get('href')
        tmp2 = page.findAll('div', class_='team2-gradient')[0].findAll('a')[0].get('href')
        link1 = f'https://www.hltv.org{tmp1}'
        link2 = f'https://www.hltv.org{tmp2}'
        arr1 = [*world_ranking_avarage_age_weeks(link1), *world_ranking_avarage_age_weeks(link2)]
        arr2 = [*last_20_matches(link1), *last_20_matches(link2)]
        arr2_tmp = []
        for i in arr2:
            for k in i:
                arr2_tmp.append(k)
        arr2 = arr2_tmp
        check_size(arr1, 10)
        check_size(arr2, counter_last_matches * 80)
        return arr1, arr2

    try:
        date = get_date(url)
        # print(date) # Perfect
        # print(*get_team_name(url)) # Perfect
        # print(get_g_score(url)) # Perfect
        # print(*get_history_score(url, date)) # Perfect
        # print(get_prize_pull(url)) # Perfect
        tmp = get_players_1(url, -1)
        arr1, arr2 = all_stat_team(url, date)
        new_list1 = [(url,
                      *get_team_name(url),
                      date,
                      get_g_score(url),
                      *get_history_score(url, date),
                      get_prize_pull(url),
                      *tmp,
                      *arr1,
                      *arr2,)
                     ]
        with open('data.csv', "a+", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(new_list1)
        # Final size = 79 + 80 * x
    except:
        pass


parse_links()
first_line = [("URL", "NAME1", "NAME2", 'DATE', 'G_SCORE (A/(A + B))',
               'HISTORY_SCORE_A', 'HISTORY_SCORE_B',
               'HISTORY_SCORE_MAPS ((SUM(A - B)/N + 16) / 32)' 'PRIZE_POOL',
               'KD x 10', 'KPR x 10', 'HEADSHOTS x 10', 'MAPS x 10',
               'DPR x 10', 'RC x 10', 'DONT TRY TO UNDERSTAND!!!!'
               )]

"""with open('data.csv', "a+", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(first_line)"""


def main(arr, i):
    for k in range(len(arr)):
        error = 0
        with open('data.csv', "r", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == f'https://www.hltv.org{arr[k]}':
                    error = 1
        if error == 0:
            print(f'Матч - {k}, поток - {i}')
            print(f'https://www.hltv.org{arr[k]}')
            get_data(f'https://www.hltv.org{arr[k]}')
    """for k in range(len(arr)):
        print(f'Матч - {k}, поток - {i}')
        print(f'https://www.hltv.org{arr[k]}')
        get_data(f'https://www.hltv.org{arr[k]}')"""


for i in range(3):
    with open(f'data/matches_links_{i}.json') as file:
        a = json.load(file)
    main(a, 1)


"""tmp = []
for i in range(0, 20):
    with open(f'data/matches_links_{i}.json') as file:
        a = json.load(file)
    tmp.append(Thread(target=main, args=(a, i)))


for i in tmp:
    i.start()
"""