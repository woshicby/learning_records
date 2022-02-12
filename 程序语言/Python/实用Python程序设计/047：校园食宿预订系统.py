n, m = map(int, input().split())
food = {}
for i in range(m):
    s = input().split()
    name, price, amount = s[0], int(s[1]), int(s[2])
    food[name] = [price, amount]
total = 0
for i in range(n):
    names = input().split()
    for name in names:
        if food[name][1] > 0:
            total += food[name][0]
            food[name][1] -= 1
print(total)
