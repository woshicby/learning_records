# 027:数字反转
n = input()
flag = 0
if n[0] in '+-':
    print(n[0], end="")
    for i in range(-1, -len(n), -1):
        if n[i] != "0":
            flag = 1
        if flag == 1:
            print(n[i], end="")
else:
    for i in range(-1, -len(n) - 1, -1):
        if n[i] != "0":
            flag = 1
        if flag == 1:
            print(n[i], end="")
