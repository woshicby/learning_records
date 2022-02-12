# 029:数字统计
count = 0
Range = input().split()
for i in range(int(Range[0]), int(Range[1]) + 1):
    for j in str(i):
        if j == "2":
            count += 1
print(count)