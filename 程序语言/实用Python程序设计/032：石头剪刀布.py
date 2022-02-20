# 032:石头剪刀布
def result(a, b):
    if a == b:
        return 0
    if a == 5 and b == 0:
        return 1
    if a == 0 and b == 5:
        return -1
    if a < b:
        return 1
    else:
        return -1


s = input().split()
n, na, nb = int(s[0]), int(s[1]), int(s[2])
sa = input().split()
sb = input().split()
winA = winB = 0
ptrA = ptrB = 0
for i in range(n):
    r = result(int(sa[ptrA]), int(sb[ptrB]))
    if r == 1:
        winA += 1
    elif r == -1:
        winB += 1
    ptrA = (ptrA + 1) % na
    ptrB = (ptrB + 1) % nb
if winA > winB:
    print("A")
elif winA < winB:
    print("B")
else:
    print("draw")
