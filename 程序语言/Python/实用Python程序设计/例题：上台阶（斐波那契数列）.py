def ways(n):
    if n == 1:
        return 1
    elif n == 2:
        return 2
    else:
        return ways(n - 1) + ways(n - 2)


for i in range(100):
    print(ways(i+1))
