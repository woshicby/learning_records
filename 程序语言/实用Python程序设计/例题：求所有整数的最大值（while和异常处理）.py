# 求所有整数的最大值（while+异常处理）
s = input()
lst = s.split()
maxV = int(lst[0])
try:
    while True:
        lst = s.split()
        for x in lst:
            maxV = max(maxV, int(x))
        s = input()
except:
    pass
print(maxV)