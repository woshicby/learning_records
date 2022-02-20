import re

m = r"(^|[^0-9-])(\d{1,2})([^0-9]|$)"
for i in range(2):
    s = input()
    lst = re.findall(m, s)
    for x in lst:
        print(x[1])
