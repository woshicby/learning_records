# 023:奥运奖牌计数
gold = silver = bronze = 0
for i in range(int(input())):
    get = input().split()
    gold += int(get[0])
    silver += int(get[1])
    bronze += int(get[2])
print(gold, silver, bronze, gold + silver + bronze)
