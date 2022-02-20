# 一次输入多个+一次赋值多个
a = input()
num = a.split()
a, b = int(num[0][0] + num[0][1]), int(num[1][0] + num[1][1])
print(a + b)
