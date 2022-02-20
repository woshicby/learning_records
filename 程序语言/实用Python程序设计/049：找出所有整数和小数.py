import re

m = r'\d+\.\d+|\d+'
while True:
    try:
        s = input()
        lst = re.findall(m, s)
        for x in lst:
            print(x)
    except:
        break
