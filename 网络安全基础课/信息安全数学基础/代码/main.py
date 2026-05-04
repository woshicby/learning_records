import time  # 时间包


# 编程实现模重复平方算法求2^1000000mod1309
def square_and_multiply(m, e, n):  # m的e次方模n
    output = 1
    bit_stack_list = []  # 本质上是用列表做栈
    while e != 0:
        e_i = e % 2  # 得二进制最低位
        bit_stack_list.append(e_i)
        e = e // 2  # 除以二的商（右移一位）
    while bit_stack_list:
        e_i = bit_stack_list.pop()
        output = output * output % n
        if e_i == 1:
            output = output * m % n
    return output


if __name__ == '__main__':
    # 课本例题验证正确性
    m, e, n = 9726, 3533, 11413
    print('=====测试模平方算法和直接pow的用时,求{0}的{1}次方模{2}====='.format(m, e, n))
    T0 = time.perf_counter()  # 开始计时
    print(square_and_multiply(m, e, n))
    T1 = time.perf_counter()
    print("模平方算法用时", (T1 - T0) * 1000, "毫秒")
    T0 = time.perf_counter()  # 开始计时
    print(pow(m, e) % n)
    T2 = time.perf_counter()
    print("pow()+mod用时", (T2 - T0) * 1000, "毫秒")
    # 用生成的一组私钥测试一下
    m, e, n = 12345678, 28351877, 37067851
    print('=====测试模平方算法和直接pow的用时,求{0}的{1}次方模{2}====='.format(m, e, n))
    T0 = time.perf_counter()  # 开始计时
    print(square_and_multiply(m, e, n))
    T1 = time.perf_counter()
    print("模平方算法用时", (T1 - T0) * 1000, "毫秒")
    T0 = time.perf_counter()  # 开始计时
    print(pow(m, e) % n)
    T2 = time.perf_counter()
    print("pow()+mod用时", (T2 - T0) * 1000, "毫秒")

# 给定一个奇素数p，输出p的平方剩余和平方非剩余
# 基于模重复平方算法，判断a是否为模p的平方剩余
