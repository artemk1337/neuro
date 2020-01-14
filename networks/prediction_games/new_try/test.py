from hltv.hltv import *
import json


"""<==========MANUAL==========>"""

"""

top5teams()
top30teams()
top_players()
get_players(teamid)
get_team_info(teamid)
get_matches()
get_results() - last 100 matchese
get_results_by_date(start_date, end_date)

"""


# print(get_matches())
a = get_team_info(9085)
# print(get_team_info(9085))

with open('check.json', 'w', encoding='utf-8') as f:
    json.dump(a, f, indent=4)


