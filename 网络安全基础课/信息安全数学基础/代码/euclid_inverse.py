def recursive_euclid(a, b):  # 使用递归的欧几里得算法
    if b == 0:
        return a
    else:
        return recursive_euclid(b, a % b)


def loop_euclid(a, b):  # 使用循环的欧几里得算法
    while b != 0:
        r = a % b
        a = b
        b = r
    return a


def recursive_extended_euclid(a, b):  # 使用递归的扩展的欧几里得算法
    r, s, t = recursive(a, b, 1, 0, 0, 1)
    return r, s, t, a, b


def recursive(r, r1, s, s1, t, t1):  # 递归操作部分
    if r1 != 0:
        q = r // r1
        temp1, temp2, temp3 = r - q * r1, s - q * s1, t - q * t1
        return recursive(r1, temp1, s1, temp2, t1, temp3)
    else:
        return r, s, t


def loop_extended_euclid(a, b):  # 使用循环的扩展的欧几里得算法
    r, r1, s, s1, t, t1 = a, b, 1, 0, 0, 1
    while r1 != 0:
        q = r // r1
        temp1, temp2, temp3 = r - q * r1, s - q * s1, t - q * t1
        r, r1, s, s1, t, t1 = r1, temp1, s1, temp2, t1, temp3
        # print(q, r, r1, s, s1, t, t1)
    return r, s, t, a, b


def inverse(a, b):  # 求逆元
    r, r1, s, s1, t, t1 = a, b, 1, 0, 0, 1
    while r1 != 0 or r1 != 1:
        q = r // r1
        temp1, temp2, temp3 = r - q * r1, s - q * s1, t - q * t1
        r, r1, s, s1, t, t1 = r1, temp1, s1, temp2, t1, temp3
        if r1 == 1:
            if t1 < 0:
                return a + t1
            else:
                return t1
        if r1 == 0:
            return "不存在"


# 输出辅助
def format_to_bezout_equation(result):  # 把输出值格式化为可读的等式
    return str(result[0]) + "=" + str(result[1]) + "×" + str(result[3]) + "+" + str(result[2]) + "×" + str(result[4])


# 测试代码
if __name__ == '__main__':
    num = input().split()
    a0, b0 = int(num[0]), int(num[1])
    print(a0, b0, "用递归的欧几里得算法求得的最大公因数为", recursive_euclid(a0, b0))
    print(a0, b0, "用循环的欧几里得算法求得的最大公因数为", loop_euclid(a0, b0))
    print(a0, b0, "用递归的扩展的欧几里得算法求得的贝祖等式为", format_to_bezout_equation(recursive_extended_euclid(a0, b0)))
    print(a0, b0, "用循环的扩展的欧几里得算法求得的贝祖等式为", format_to_bezout_equation(loop_extended_euclid(a0, b0)))
    if a0 < b0:  # 保证a>b（虽然反着来也能算）
        num = a0
        a0 = b0
        b0 = num
    print(b0, "模", a0, "的乘法逆元为", inverse(a0, b0))
