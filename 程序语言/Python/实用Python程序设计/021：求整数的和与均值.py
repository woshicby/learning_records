# 021:求整数的和与均值
Sum = 0
n = int(input())
for i in range(n):
    Sum += int(input())
print(str(Sum), "%.5f" % (Sum / n))
