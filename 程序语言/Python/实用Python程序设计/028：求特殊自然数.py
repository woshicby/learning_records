# 028:求特殊自然数
for i in range(81, 343):
    x = i
    # 求七进制
    str71 = str(x % 7)
    x //= 7
    str72 = str(x % 7)
    x //= 7
    str73 = str(x % 7)
    # 求九进制
    x = i
    str91 = str(x % 9)
    x //= 9
    str92 = str(x % 9)
    x //= 9
    str93 = str(x % 9)
    if str71 == str93 and str72 == str92 and str73 == str91:
        print(i)
        print(str73 + str72 + str71)
        print(str93 + str92 + str91)
