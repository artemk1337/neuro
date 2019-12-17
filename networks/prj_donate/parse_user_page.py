import vk
import json
import random


public_name = ['wgcsgo', 'leagueoflegends', 'fortnite', 'dota2', 'worldofwarcraft']
fn = 'worldofwarcraft'
token = "5eda4a2a5eda4a2a5eda4a2ae95eb4eeef55eda5eda4a2a032346f2f5ec8e77246de2bb"  # Сервисный ключ доступа


session = vk.Session(access_token=token)  # Авторизация
vk_api = vk.API(session)


for fn in public_name:
    with open(f'data/{fn}.txt', 'r') as f:
        data = f.readlines()

    arr = []
    for i in data:
        tmp = i.split('id')[1]
        tmp = int(tmp.split('\n')[0])
        arr.append(tmp)

    random.shuffle(arr)
    final = {}
    counter = 0
    for i in range(len(arr)):
        id = arr[i]
        max_persons = 1000
        try:
            user = vk_api.users.get(user_ids=id, v=5.92,
                                    fields=['sex', 'bdate', 'city', 'country', 'home_town', 'photo_id'])[0]
            if len(user['bdate'].split('.')) == 3 and user['city']['id'] == 1:  # Check age and city
                sub = vk_api.users.getSubscriptions(user_id=id, v=5.92)
                wall = vk_api.wall.get(owner_id=user['id'], v=5.92)
                d = {'user': user,
                     'wall': wall,
                     'sub': sub}
                final[id] = d
                if counter >= max_persons:
                    break
                print('Counter -', counter)
                counter += 1
                with open(f'data/{fn}.json', 'w') as f:
                    json.dump(final, f, separators=(',', ':'), indent=4)
        except Exception:
            # print('Error')
            pass





