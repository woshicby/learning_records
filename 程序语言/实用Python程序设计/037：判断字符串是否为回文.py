# 037:判断字符串是否为回文
stringA = input()
if stringA[::-1] == stringA:
    print('yes')
else:
    print('no')
