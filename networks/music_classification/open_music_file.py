
def get_total_time(heroes, n):
    import random
    arr = [0 for i in range(0, n)]
    heroes.sort()
    print(heroes)
    for i in range(len(heroes) ** 2):
        arr = [0 for i in range(0, n)]
        cpy = [i for i in heroes]
        while len(cpy) > 0:
            min_ = min(arr)
            index = arr.index(min_)
            k = random.randint(0, len(cpy) - 1)
            index1 = cpy.index(k)
            arr[index] += cpy[index1]
            del cpy[index1]
    return sum(arr)


print(get_total_time([6, 7, 4, 1, 3], 2))

