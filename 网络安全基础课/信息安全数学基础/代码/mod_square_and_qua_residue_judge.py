import time  # 时间包
from euclid_inverse import recursive_euclid  # 导入最大公因数
from find_mersenne import is_prime  # 导入判断素数


# 编程实现模重复平方算法求2^1000000mod1309
def square_and_multiply(m, e, n):  # m的e次方模n
    output = 1
    bit_stack_list = []  # 本质上是用列表做栈
    while e != 0:
        e_i = e % 2  # 得二进制最低位
        bit_stack_list.append(e_i)  # 入栈！
        e = e // 2  # 除以二的商（右移一位）
    while bit_stack_list:
        e_i = bit_stack_list.pop()  # 出栈！
        output = output * output % n
        if e_i == 1:
            output = output * m % n
    return output


# 基于模重复平方算法，判断a是否为模p的平方剩余
def euler_quad_residue_judge(a, p):
    # 需要a、p互素，p是素数
    if not is_prime(p):
        return 0
    ex = square_and_multiply(a, (p - 1) / 2, p)
    if ex == 1:
        return 1
    else:
        return -1


# 给定一个奇素数p，输出p的平方剩余和平方非剩余
def quad_residue_list_euler(p):
    # p是素数
    if not is_prime(p):
        return 0, 0
    else:
        quad_residue_list = []
        quad_non_residue_list = []
        for i in range(1, p):
            if euler_quad_residue_judge(i, p) == 1:
                quad_residue_list.append(i)
            elif euler_quad_residue_judge(i, p) == -1:
                quad_non_residue_list.append(i)
        return quad_residue_list, quad_non_residue_list


if __name__ == '__main__':
    # 课本例题验证正确性
    m, e, n = 9726, 3533, 11413
    print('=====测试模平方算法和直接pow的用时,求{0}的{1}次方模{2}====='.format(m, e, n))
    T0 = time.perf_counter()  # 开始计时（覆盖原打点）
    print(square_and_multiply(m, e, n))
    T1 = time.perf_counter()
    print("模平方算法用时", (T1 - T0) * 1000, "毫秒")
    T0 = time.perf_counter()  # 开始计时（覆盖原打点）
    print(pow(m, e) % n)
    T2 = time.perf_counter()
    print("pow()+mod用时", (T2 - T0) * 1000, "毫秒")

    # 用前面生成的一组私钥测试一下
    m, e, n = 12345678, 28351877, 37067851
    print('=====测试模平方算法和直接pow的用时,求{0}的{1}次方模{2}====='.format(m, e, n))
    T0 = time.perf_counter()  # 开始计时（覆盖原打点）
    print(square_and_multiply(m, e, n))
    T1 = time.perf_counter()
    print("模平方算法用时", (T1 - T0) * 1000, "毫秒")
    T0 = time.perf_counter()  # 开始计时（覆盖原打点）
    print(pow(m, e) % n)
    T2 = time.perf_counter()
    print("pow()+mod用时", (T2 - T0) * 1000, "毫秒")
    # 测试平方剩余非剩余
    test_list = [97, 98, 997]  # 小质数，小合数，（稍微）大质数
    for i in test_list:
        residue_list, non_residue_list = quad_residue_list_euler(i)
        if residue_list == 0:
            print(i, "是合数，暂不计算")
        else:
            print("模", i, "的平方剩余有：", residue_list, "\n模", i, "的非平方剩余有：", non_residue_list)
