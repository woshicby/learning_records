# 018:大象喝水
a = input()
b = int(a.split()[1])
a = int(a.split()[0])
per = 3.14159 * b * b * a
if 20000 % per == 0:
    print(int(20000 // per))
else:
    print(int(20000 // per + 1))
