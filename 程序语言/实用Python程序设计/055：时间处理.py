import datetime
for i in range(5):
    tmstr = input()
    if 'M' in tmstr:
        tm = datetime.datetime.strptime(tmstr, '%m-%d-%Y %H:%M %p')
    else:
        tm = datetime.datetime.strptime(tmstr, '%Y %m %d %H %M')
    deltastr = input()
    if ' ' in deltastr:
        deltastr = deltastr.split()
        delta = datetime.timedelta(days=float(deltastr[0]), hours=float(deltastr[1]), minutes=float(deltastr[2]))
    else:
        delta = datetime.timedelta(seconds=float(deltastr))
    tm += delta
    print(tm.strftime('%Y-%m-%d %H:%M:%S'))
