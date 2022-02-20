# 011:计算多项式的值
a = input()
num = a.split()
print("%.7f" % (float(num[1]) * float(num[0]) ** 3 + float(num[2]) * float(num[0]) ** 2 + float(num[3]) * float(num[0]) + float(num[4])))
