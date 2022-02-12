n = int(input())
a = []
for i in range(n):
    s = input().split()
    a.append((s[0], int(s[1])))
a.sort(key=lambda y: (-y[1], y[0]))
for x in a:
    print(x[0], x[1])
