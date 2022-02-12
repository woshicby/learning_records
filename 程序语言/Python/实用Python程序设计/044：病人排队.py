n = int(input())
a = []
for i in range(n):
    s = input().split()
    a.append(s)


def f(x):
    if int(x[1]) >= 60:
        return -int(x[1]), int(x[2])
    else:
        return 0, x[2]


a.sort(key=f)
for j in range(n):
    print(a[j][0])
