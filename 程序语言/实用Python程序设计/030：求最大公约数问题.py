# 030:求最大公约数问题
def gcd(a, b):
    if a % b == 0:
        return b
    else:
        return gcd(b, a % b)


num = input().split()
print(gcd(int(num[0]), int(num[1])))
