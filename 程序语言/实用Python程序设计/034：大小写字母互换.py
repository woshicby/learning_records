# 034:大小写字母互换
for c in input():
    if 'a' <= c <= 'z':
        print(chr(ord(c) - 32), end="")
    elif 'A' <= c <= 'Z':
        print(chr(ord(c) + 32), end="")
    else:
        print(c, end="")
