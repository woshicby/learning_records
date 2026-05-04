import math  # 数学包
import time  # 时间包


def is_prime(number):  # 判断：是素数
    sqrt_root = int(math.sqrt(number))
    if sqrt_root < 2:
        return 0
    for test_i in range(2, sqrt_root + 1):
        if number % test_i == 0:
            return 0  # 不是素数
    return 1  # 是素数


if __name__ == '__main__':
    # 找前n个梅森素数，并打印寻找时间
    n = int(input("找几个梅森素数？"))
    T0 = time.perf_counter()  # 开始计时
    count, index = 0, 1  # 初始化计数、梅森素数的指数
    while count < n:
        testMersenne = (2 ** index) - 1
        # print(testMersenne,is_prime(testMersenne))
        if is_prime(testMersenne):
            T1 = time.perf_counter()
            print("找到一个梅森素数", testMersenne, "其指数为", index, "累计用时", (T1 - T0) * 1000, "毫秒")
            count += 1
        index += 1
