import re

m = '^[a-zA-Z][\\w-]{7,}$'

while True:
    try:
        s = input()
        if re.match(m, s) is not None:
            print("yes")
        else:
            print("no")
    except:
        break
