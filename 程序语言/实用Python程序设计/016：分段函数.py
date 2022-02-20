# 016:分段函数
x = float(input())
if 0 <= x < 5:
    print("%.3f" % (2.5 - x))
elif 5 <= x < 10:
    print("%.3f" % (2 - 1.5 * (x - 3) * (x - 3)))
else:
    print("%.3f" % (x / 2 - 1.5))
