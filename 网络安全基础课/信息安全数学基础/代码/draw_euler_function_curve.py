import matplotlib.pyplot as plt


# 每一个数都用欧几里得算法验证，想想都过于繁琐。
# 可以逐个排除有n的各种因子的数
def euler_ver1(n):
    ans = n  # 对1~n的的所有数进行判断
    i = 2
    while i <= n:  # 尝试因数i
        if n % i == 0:  # 如果n有i这个因数
            ans -= ans // i  # 排除有i这个因数的
            # n在后续步骤被改变了，只能用ans//i
            while n % i == 0:  # 除去n中的i因数
                n //= i
        i += 1  # 试试下一个数
    return ans


# 其实只要测试到sqrt(n)以下的因子就好了（和验证素数一样）
def euler(n):
    ans = n  # 对1~n的的所有数进行判断
    i = 2
    while i * i <= n:  # 尝试因数i
        if n % i == 0:  # 如果n有i这个因数
            ans -= ans // i  # 排除有i这个因数的
            # n在后续步骤被改变了，只能用ans//i
            while n % i == 0:  # 除去n中的i因数
                n //= i
        i += 1  # 试试下一个数
    if n > 1:  # 最后剩下一个素数
        ans -= ans // n  # 再排除一次
    return ans

if __name__ == '__main__':
    # 其实每个整数数都能分解成一堆素数的乘积（学过的定理）
    # 应该只要验证sqrt(n)以下的素数就好了
    # 但是还要先生成素数列表，每次求euler都生成一下有点过于麻烦
    x = []
    y = []
    base = int(input("求多少的次方数的欧拉（输入1则为求一段连续数的）"))
    head = int(input("从底数的多少次方/多少开始求欧拉"))
    tail = int(input("求欧拉求到底数的多少次方/多少")) + 1
    if base == 1:
        for num in range(head, tail):
            # print(n, euler(n))
            x.append(num)
            y.append(euler(num))
            print("求得", num, "的欧拉函数值为", euler(num))
    else:
        for exponential in range(head, tail):
            # print(n, euler(n))
            x.append(base ** exponential)
            y.append(euler(base ** exponential))
            print("求得", base, "的", exponential, "次幂", base ** exponential, "的欧拉函数值为", euler(base ** exponential))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    plt.scatter(x, y, s=3)
    plt.plot(x, y, color='r')
    plt.xlabel("n")
    plt.ylabel("φ(n)")
    plt.show()
