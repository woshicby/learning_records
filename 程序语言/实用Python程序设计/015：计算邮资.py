# 015:计算邮资
a = input()
b = a.split()[1]
a = int(a.split()[0])
if a < 1000:
    p = 8
else:
    p = 8 + ((a - 1001) // 500 + 1) * 4
if b == 'y':
    p += 5
print(p)
