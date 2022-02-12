# 038:字符串最大跨距
strings = input().split(',')
if strings[0].rfind(strings[1]) + len(strings[1]) <= strings[0].rfind(strings[2]):
    # print(strings[0].find(strings[1]))
    # print(strings[0].rfind(strings[2]))
    # print(strings[0].find(strings[1])+len(strings[1]))
    print(strings[0].rfind(strings[2]) - len(strings[1]) - strings[0].find(strings[1]))
else:
    print('-1')
