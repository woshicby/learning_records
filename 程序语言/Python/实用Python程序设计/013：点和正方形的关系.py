# 013:点和正方形的关系
a = input()
b = int(a.split()[1])
a = int(a.split()[0])
if -1 <= a <= 1 and -1 <= b <= 1:
    print("yes")
else:
    print("no")
