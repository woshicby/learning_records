for i in {2, 3}:
    flag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for j in range(16):
        print(i, "的", j + 1, "次方模17为", i ** (j + 1) % 17)
        flag[i ** (j + 1) % 17-1] = 1
    for j in range(16):
        if flag[j] == 0:
            print("没有遍历到", j+1)
    print("\n")