# 022:整数序列的元素最大跨度值
maxV = 0  # 存储max的数
minV = 1001  # 存储min的数
num = input()  # 略过个数
num = input().split()  # 读取数组
for x in num:
    maxV = max(maxV, int(x))
    minV = min(minV, int(x))
print(maxV - minV)
