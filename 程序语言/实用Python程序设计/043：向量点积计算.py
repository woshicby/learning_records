n = int(input())
s = list(map(int, input().split()))
t = list(map(int, input().split()))
a = []
total = 0
for i in range(n):
    a.append(s[i] * t[i])
    total += a[i]
print(total)
