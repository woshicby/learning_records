m, n, k = map(int, input().split())
a = []
b = []
d = []
for j in range(k):
    d.append(0)
c = [d] * m
total = 0
for i in range(m):
    s = list(map(int, input().split()))
    a.append(s)
for i in range(n):
    t = list(map(int, input().split()))
    b.append(t)
for i in range(m):
    for j in range(k):
        for p in range(n):
            total += a[i][p] * b[p][j]
        c[i][j] = total
        print(c[i][j], end=' ')
        total = 0
    print('')
