# 017:简单计算器
a = input()
c = a.split()[2]
b = int(a.split()[1])
a = int(a.split()[0])
if c == '/':
    if b == 0:
        print("Divided by zero!")
    else:
        print(a // b)
elif c in "+-*":
    print(eval(str(a) + c + str(b)))
else:
    print("Invalid operator!")
