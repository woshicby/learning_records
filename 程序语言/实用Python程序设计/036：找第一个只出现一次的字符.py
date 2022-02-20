# 036:找第一个只出现一次的字符
def find(string):
    for i in range(len(string)):
        if not string[i] in string[:i] and not string[i] in string[i+1:]:
            print(string[i])
            return
    print("no")
    return


find(input())
