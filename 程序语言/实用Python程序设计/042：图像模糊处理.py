import copy

n, m = map(int, input().split())
a = []
for i in range(n):
    lst = list(map(int, input().split()))
    a.append(lst)
b = copy.deepcopy(a)  # b = a[:] 是浅拷贝，不行
for i in range(1, n - 1):
    for j in range(1, m - 1):
        b[i][j] = round((a[i][j] + a[i - 1][j] + a[i + 1][j] + a[i][j - 1] + a[i][j + 1]) / 5)
for i in range(0, n):
    for j in range(0, m):
        print(b[i][j], end=" ")
    print("")
