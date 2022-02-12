# 024:鸡尾酒疗法
n = int(input())
case = input().split()
cocktail_result = int(case[1]) / int(case[0])
for i in range(n - 1):
    case = input().split()
    test_result = int(case[1]) / int(case[0])
    if test_result - cocktail_result > 0.05:
        print("better")
    elif cocktail_result - test_result > 0.05:
        print("worse")
    else:
        print("same")
