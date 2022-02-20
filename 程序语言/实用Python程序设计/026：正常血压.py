# 026:正常血压
count = count_max = 0
n = int(input())
for i in range(n):
    pressure = input().split()
    if 90 <= int(pressure[0]) <= 140 and 60 <= int(pressure[1]) <= 90:
        count += 1
        count_max = max(count_max, count)
    else:
        count = 0
print(count_max)
