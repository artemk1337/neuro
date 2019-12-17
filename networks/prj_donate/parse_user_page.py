import vk
import json


token = "9d480cbc9d480cbc9d480cbc089d26ae6399d489d480cbcc0b0b00ca99b5d870cbf1cb7"  # Сервисный ключ доступа
session = vk.Session(access_token=token)  # Авторизация
vk_api = vk.API(session)

user = vk_api.users.get(user_ids='eridi96', v=5.92,
                       fields=['sex', 'bdate', 'city', 'country', 'home_town', 'photo_id'])[0]
print(user)
# quit()
a = vk_api.wall.get(owner_id=user['id'], v=5.92)
print(len(a['items']))
quit()
with open('data/parse.json', 'w') as f:
    json.dump(a, f, separators=(',', ':'), indent=4)








