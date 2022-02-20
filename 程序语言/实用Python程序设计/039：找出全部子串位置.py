# 039:找出全部子串位置
for i in range(int(input())):
    strings = input().split()
    if not strings[1] in strings[0]:
        print("no")
    else:
        j = 0
        while j < len(strings[0]) and strings[0].find(strings[1], j) != -1:
            print(strings[0].find(strings[1], j), end=" ")
            j = strings[0].find(strings[1], j) + len(strings[1])
        print("")
