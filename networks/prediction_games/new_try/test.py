import re
import requests
import datetime
from bs4 import BeautifulSoup
from python_utils import converters
import pprint
import sqlite3
from contextlib import closing


pp = pprint.PrettyPrinter()


def get_parsed_page(url):
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    return BeautifulSoup(requests.get(url, headers=headers).text, "lxml")


with closing(sqlite3.connect('teams_info.db')) as conn:
    c = conn.cursor()
    c.execute('''CREATE TABLE hltv(
    last_update_info integer,
    team text,
    link text,
    rank integer,
    current_win_streak integer,
    win_rate real,
    last_matches blob,
    stat_player)''')
    conn.commit()








pp.pprint(get_parsed_page("https://betscsgo.top/history"))






