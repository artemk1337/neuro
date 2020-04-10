number = 6

target1 = str('{0}{1}.{1}'.format(number, (number) % 6 + 1))
target2 = str('{1}{0}.{1}'.format(number, (number - 2) % 6 + 1))
print('en\nconf t')

print(
f"int f0/0\n\
ip address 172.16.{number}.2 255.255.255.0\n\
no shutdown\n\
int s2/0\n\
ip address 172.16.{target1.split('.')[0]}.{number} 255.255.255.0\n\
clock rate 1200\n\
no shutdown\n\
int s3/0\n\
ip address 172.16.{target2.split('.')[0]}.{number} 255.255.255.0\n\
clock rate 1200\n\
no shutdown\n\
end\n\
"
)
print('en\nconf t')


arr = [1, 2, 3, 4, 5, 6]


counter = 0
pos = number
while arr[(pos + 1) % 6] != number:
    if counter >= 2:
        print(f'ip route 172.16.{arr[pos % 6]}{arr[(pos + 1) % 6]}.0 255.255.255.0 172.16.{arr[(number - 2) % 6]}{arr[(number - 1) % 6]}.{arr[(number - 2) % 6]}')
    else:
        print(f'ip route 172.16.{arr[pos % 6]}{arr[(pos + 1) % 6]}.0 255.255.255.0 172.16.{arr[(number - 1) % 6]}{arr[(number) % 6]}.{arr[(number) % 6]}')
    pos += 1
    counter += 1


pos = number - 1
counter = 1
while counter <= 3:
    print(f'ip route 172.16.{arr[(pos - counter) % 6]}.0 255.255.255.0 172.16.{arr[(number - 2) % 6]}{arr[(number - 1) % 6]}.{arr[(number - 2) % 6]}')
    print(f'ip route 172.16.{arr[(pos + counter) % 6]}.0 255.255.255.0 172.16.{arr[(number - 1) % 6]}{arr[(number) % 6]}.{arr[(number) % 6]}')
    counter += 1


