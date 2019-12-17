import vk
import json


token = "9d480cbc9d480cbc9d480cbc089d26ae6399d489d480cbcc0b0b00ca99b5d870cbf1cb7"  # Сервисный ключ доступа
session = vk.Session(access_token=token)  # Авторизация
vk_api = vk.API(session)

with open('data/dota2.txt', 'r') as f:
    data = f.readlines()

arr = []
for i in data:
    tmp = i.split('id')[1]
    tmp = int(tmp.split('\n')[0])
    arr.append(tmp)

print(arr)

final = {}

for i in range(len(arr)):
    id = arr[i]
    max_persons = 1000
    try:
        user = vk_api.users.get(user_ids=id, v=5.92,
                                fields=['sex', 'bdate', 'city', 'country', 'home_town', 'photo_id'])[0]
        sub = vk_api.users.getSubscriptions(user_id=id, v=5.92)
        wall = vk_api.wall.get(owner_id=user['id'], v=5.92)
        d = {'user': user,
             'wall': wall,
             'sub': sub}
        final['id'] = id
        final['data'] = d
    except Exception:
        pass
    if i >= max_persons:
        break


with open('data/parse.json', 'w') as f:
    json.dump(final, f, separators=(',', ':'), indent=4)



