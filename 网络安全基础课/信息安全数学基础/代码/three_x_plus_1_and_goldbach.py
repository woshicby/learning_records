# 编程验证3x+1猜想，并输出归一步数
from find_mersenne import is_prime


def prime_array_gen(up_limit, button_limit=2):  # 生成素数列表用于抽取
    prime_array = []
    for i in range(button_limit, up_limit):
        if is_prime(i):
            prime_array.append(i)
    return prime_array


def one_step(number):  # 一步操作
    if number % 2 == 0:
        return number / 2
    else:
        return 3 * number + 1


if __name__ == '__main__':
    n = int(input("\n对多少及以下的数验证3x+1猜想？"))
    for num in range(n):
        count = 0  # 初始化步数
        test_num = num + 1
        while test_num != 1:
            test_num = one_step(test_num)
            count += 1
        print(str(num + 1) + "归1要" + str(count) + "步", end=",")
    print("都符合3x+1猜想")
    # 编程验证哥德巴赫猜想
    n = int(input("\n对多少及以下的偶数验证哥德巴赫猜想？（需大于4，否则无意义）"))
    prime = prime_array_gen(n + 1)  # 生成质数数组
    a, b = 0, 0
    print("小于等于", n, "的质数有", prime)
    for num in range(4, n + 1):
        if num % 2 == 0:
            for a in prime:
                for b in prime:
                    if num == a + b:
                        print(str(num) + "=" + str(a) + "+" + str(b), end=",")
                        break  # 找到了，跳出for b循环体
                if num == a + b:
                    break  # 找到了，跳出for a循环
            if num != a + b:
                print(str(num) + "找不到一对质数的和与之相等\n哥德巴赫猜想是错的")
                break  # 找到反例，跳出for num循环体
    else:
        print("都符合哥德巴赫猜想")
