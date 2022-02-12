# 014:三角形判断
a = input()
c = int(a.split()[2])
b = int(a.split()[1])
a = int(a.split()[0])
if a + b > c and a + c > b and b + c > a:
    print("yes")
else:
    print("no")
