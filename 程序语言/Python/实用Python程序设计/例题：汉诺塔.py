def hanoi(n, src, mid, des):
    # scr>>des,use mid
    if n == 1:
        print(src + "->" + des)
        return
    hanoi(n - 1, src, des, mid)
    print(src + "->" + des)
    hanoi(n - 1, mid, src, des)


hanoi(int(input()), "A", "B", "C")
