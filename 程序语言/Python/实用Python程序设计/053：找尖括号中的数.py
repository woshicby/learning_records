import re


def f(s):
    m = '\\<(0|[1-9]\\d{0,2})\\>'
    lst = re.findall(m, s)
    if not lst:
        print('NONE')
    else:
        print(*lst)


n = int(input())
for i in range(n):
    f(input())
