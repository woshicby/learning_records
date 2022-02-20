# 019:苹果和虫子2
a = input()
c = int(a.split()[2])
b = int(a.split()[1])
a = int(a.split()[0])
if a - c / b > 0:
    print(int(a - c / b))
else:
    print(0)
