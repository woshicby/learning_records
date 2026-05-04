from mod_square_and_qua_residue_judge import euler_quad_residue_judge  # 导入欧拉平方剩余判定
from mod_square_and_qua_residue_judge import square_and_multiply  # 导入模平方算法
import time  # 时间包
import matplotlib.pyplot as plt  # 画图包


def legendre_definition(a, p):  # 按定义求解勒让德符号
    if p % a == 0:
        return 0
    elif euler_quad_residue_judge(a, p) == 1:
        return 1
    elif euler_quad_residue_judge(a, p) == -1:
        return -1


def legendre_euler(a, p):  # 欧拉判定法则求解勒让德符号
    output = square_and_multiply(a, (p - 1) // 2, p)
    if output - p == -1:
        return -1
    else:
        return square_and_multiply(a, (p - 1) // 2, p)


def spit_all_2(num):  # 把一个数字里的2完全拆出，返回去除所有2的结果和2的重数
    index = 0
    while num % 2 == 0:
        num //= 2
        index += 1
    return num, index


def count_bigger_then_half_p(a, p):  # 数最小剩余系里比p/2大的
    min_residue_list = []
    count = 0
    for i in range(1, (p + 1) // 2):
        min_residue_list.append(a * i % p)
    for i in min_residue_list:
        if p / 2 < i:
            count += 1
    return count


def legendre_gauss_theorem(a, p):  # 高斯定理求解勒让德符号
    s = count_bigger_then_half_p(a, p)
    return pow(-1, s)


def legendre_gauss_lemma(a, p):  # 高斯引理求解勒让德符号
    t = count_bigger_then_half_p(a, p) % 2
    return pow(-1, t)


def legendre_l_fast(a, p):  # 二次互反律求解勒让德符号
    if a == 0:  # 为0跳出
        return 0
    elif a == 1:  # 为1跳出
        return 1
    else:
        a1, e = spit_all_2(a)
    if e % 2 == 0:
        legendre = 1
    elif p % 8 == 1 or p % 8 == 7:
        legendre = 1
    elif p % 8 == 3 or p % 8 == 5:
        legendre = -1
    if p % 4 == 3 and a1 % 4 == 3:
        legendre *= -1
    p1 = p % a1
    if a1 == 1:
        return legendre
    else:
        # print(p1, a1, legendre)
        return legendre * legendre_l_fast(p1, a1)


if __name__ == '__main__':
    p = 97
    a_list = [13681, 13687, 13691, 13693, 13697, 13709, 13711, 13721, 13723, 13729, 13751, 13757, 13759, 13763, 13781, 13789, 13799, 13807, 13829, 13831, 13841, 13859, 13873, 13877, 13879, 13883, 13901, 13903, 13907, 13913, 13921, 13931, 13933, 13963, 13967, 13997, 13999, 14009, 14011, 14029, 14033, 14051, 14057, 14071, 14081, 14083, 14087, 14107, 14143, 14149, 14153, 14159, 14173, 14177, 14197, 14207, 14221, 14243, 14249, 14251, 14281, 14293, 14303, 14321, 14323, 14327, 14341, 14347, 14369,
              14387, 14389, 14401, 14407, 14411, 14419, 14423, 14431, 14437, 14447, 14449, 14461, 14479, 14489, 14503, 14519, 14533, 14537, 14543, 14549, 14551, 14557, 14561, 14563, 14591, 14593, 14621, 14627, 14629, 14633, 14639, 14653, 14657, 14669, 14683, 14699, 14713, 14717, 14723, 14731, 14737, 14741, 14747, 14753, 14759, 14767, 14771, 14779, 14783, 14797, 14813, 14821, 14827, 14831, 14843, 14851, 14867, 14869, 14879, 14887, 14891, 14897, 14923, 14929, 14939, 14947, 14951, 14957, 14969,
              14983, 15013, 15017, 15031, 15053, 15061, 15073, 15077, 15083, 15091, 15101, 15107]  # 取了一段素数
    definition_time_list = []
    euler_time_list = []
    gauss_theorem_time_list = []
    gauss_lemma_time_list = []
    l_fast_time_list = []
    for a in a_list:
        print("=====计算a={0},p={1}的勒让德符号=====".format(a, p))
        T0 = time.perf_counter()  # 开始计时（覆盖原打点）
        print("按定义求得legendre({0},{1})的结果为{2}".format(a, p, legendre_definition(a, p)))
        T1 = time.perf_counter()
        print("按欧拉判定法则求得legendre({0},{1})的结果为{2}".format(a, p, legendre_euler(a, p)))
        T2 = time.perf_counter()
        print("按高斯定理求得legendre({0},{1})的结果为{2}".format(a, p, legendre_gauss_theorem(a, p)))
        T3 = time.perf_counter()
        print("按高斯引理求得legendre({0},{1})的结果为{2}".format(a, p, legendre_gauss_lemma(a, p)))
        T4 = time.perf_counter()
        print("按二次互反律求得legendre({0},{1})的结果为{2}".format(a, p, legendre_l_fast(a, p)))
        T5 = time.perf_counter()
        # 记录用时
        definition_time_list.append((T1 - T0) * 1000)
        euler_time_list.append((T2 - T1) * 1000)
        gauss_theorem_time_list.append((T3 - T2) * 1000)
        gauss_lemma_time_list.append((T4 - T3) * 1000)
        l_fast_time_list.append((T5 - T4) * 1000)
        print("用时分别为：{0}ms、{1}ms、{2}ms、{3}ms、{4}ms".format((T1 - T0) * 1000, (T2 - T1) * 1000, (T3 - T2) * 1000, (T4 - T3) * 1000, (T5 - T4) * 1000))
    # 画图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    plt.scatter(a_list, definition_time_list, s=3)
    plt.plot(a_list, definition_time_list, label="按定义")
    plt.scatter(a_list, euler_time_list, s=3)
    plt.plot(a_list, euler_time_list, label="欧拉法则")
    plt.scatter(a_list, gauss_theorem_time_list, s=3)
    plt.plot(a_list, gauss_theorem_time_list, label="高斯定理")
    plt.scatter(a_list, gauss_lemma_time_list, s=3)
    plt.plot(a_list, gauss_lemma_time_list, label="高斯引理")
    plt.scatter(a_list, l_fast_time_list, s=3)
    plt.plot(a_list, l_fast_time_list, label="二次互反")
    plt.xlabel("a的值（p=97）")
    plt.ylabel("用时")
    plt.legend(loc="best")
    plt.show()
