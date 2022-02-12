# 025:角谷猜想
n = int(input())
while n != 1:
    if n % 2:
        m = n
        n = n * 3 + 1
        print(str(m) + "*3+1=" + str(n))
    else:
        m = n
        n = n // 2
        print(str(m) + "/2=" + str(n))
print("End")
