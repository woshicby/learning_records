# 031:多少种取法
def ways(m, n, s):
    if m < n:
        return 0
    elif m == 0 and (n != 0 or s != 0):
        return 0
    elif m == n == s == 0:
        return 1
    elif m > s:
        return ways(s, n, s)
    else:
        return ways(m - 1, n - 1, s - m) + ways(m - 1, n, s)


time = int(input())
for i in range(time):
    num = input().split()
    print(ways(int(num[0]), int(num[1]), int(num[2])))
