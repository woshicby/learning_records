# 033:统计数字字符个数
countNum = 0
stringA = input()
for i in range(len(stringA)):
    if 47 < ord(stringA[i]) < 58:
        countNum += 1
print(countNum)

