import numpy
# #####设置区域#####
TrainsFile = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch06\testSetRBF.txt'
TestFile = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch06\testSetRBF2.txt'


# #####函数定义区域#####
# #####以下是没有变化的函数#####
def load_data_set(file_name):  # 读入数据组
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def clip_alpha(aj, high, low):  # 大于high就把aj调成high，小于low就把aj调成low
    if aj > high:
        aj = high
    if low > aj:
        aj = low
    return aj


def select_j_rand(i, m):  # i：第一个alpha的下标；m：所有alpha的数目
    j = i  # 要选一个和i不一样的j
    while j == i:
        j = int(numpy.random.uniform(0, m))  # 相等就重新抽
    return j


def select_j(i, opt_struct, error_i):  # 选择第二个alpha/内循环的alpha值 -heurstic, and calcs error_j
    max_k = -1  # 初始化最大位置k为-1
    max_delta_error = 0  # 初始化最大delta_error为0
    error_j = 0  # 初始化返回的error_j=0
    opt_struct.eCache[i] = [1, error_i]  # 设置有i的有效标志位 #choose the alpha that gives the maximum delta E
    valid_ecache_list = numpy.nonzero(opt_struct.eCache[:, 0].A)[0]  # 筛选eCache第0维里不是零的（即有效的），返回他们的索引值的第0维度
    if (len(valid_ecache_list)) > 1:  # 已经有非零的值了（不是第一轮选择）
        for k in valid_ecache_list:  # 遍历有效的Ecache值，找到让delta_error最大的那个
            if k == i:  # k=i，过。
                continue
            error_k = calc_error_k(opt_struct, k)  # 算一下error_k
            delta_error = abs(error_i - error_k)  # 求error_k和error_i的差值
            if delta_error > max_delta_error:  # 更大，更新
                max_k = k  # 更新
                max_delta_error = delta_error  # 更新最大值
                error_j = error_k  # 更新返回值
        return max_k, error_j
    else:  # #没有非零的值（是第一轮选择）我们没有可用的eCache值，随机选择
        j = select_j_rand(i, opt_struct.m)
        error_j = calc_error_k(opt_struct, j)
    return j, error_j


def update_error_k(opt_struct, k):  # 每次alpha改变了就要更新他的误差缓存
    error_k = calc_error_k(opt_struct, k)
    opt_struct.eCache[k] = [1, error_k]


# #####以上是没有变化的函数#####
# #####以下是有变化的函数#####
def kernel_trans(x, a, k_tuple):  # 计算核函数/把数据映射到高维空间
    m, n = numpy.shape(x)
    k = numpy.mat(numpy.zeros((m, 1)))
    if k_tuple[0] == 'lin':  # 线性核函数
        k = x * a.T
    elif k_tuple[0] == 'rbf':  # 径向基函数
        for j in range(m):
            delta_row = x[j, :] - a
            k[j] = delta_row * delta_row.T
        k = numpy.exp(k / (-1 * k_tuple[1] ** 2))  # 逐个元素相除
    else:
        raise NameError('核函数无法识别')
    return k


class OptStruct:  # 增加了.K参数
    def __init__(self, data_mat_in, class_labels, c, tolerate, k_tuple):  # 用参数来初始化结构体
        self.X = data_mat_in
        self.labelMat = class_labels
        self.C = c
        self.tol = tolerate
        self.m = numpy.shape(data_mat_in)[0]
        self.alphas = numpy.mat(numpy.zeros((self.m, 1)))
        self.b = 0
        self.eCache = numpy.mat(numpy.zeros((self.m, 2)))  # 第一栏是有效标志符
        self.K = numpy.mat(numpy.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], k_tuple)


def calc_error_k(opt_struct, k):  # f_xk的计算有改动
    f_xk = float(numpy.multiply(opt_struct.alphas, opt_struct.labelMat).T * opt_struct.K[:, k] + opt_struct.b)
    error_k = f_xk - float(opt_struct.labelMat[k])
    return error_k


def inner_loop(i, opt_struct):  # eta的计算变成了用核函数的版本
    error_i = calc_error_k(opt_struct, i)
    # opt_struct.tol：容忍错误的程度
    # labelMat[i]*Ei < -opt_struct.tol，且不>=C，则alphas[i]要增大
    # labelMat[i]*Ei > opt_struct.tol，且不<=0,则alphas[i]要减小
    if ((opt_struct.labelMat[i] * error_i < -opt_struct.tol) and (opt_struct.alphas[i] < opt_struct.C)) or ((opt_struct.labelMat[i] * error_i > opt_struct.tol) and (opt_struct.alphas[i] > 0)):
        j, error_j = select_j(i, opt_struct, error_i)  # 第一次选择，结果是用select_j_rand产生的
        alpha_i_old = opt_struct.alphas[i].copy()
        alpha_j_old = opt_struct.alphas[j].copy()
        if opt_struct.labelMat[i] != opt_struct.labelMat[j]:
            low = max(0, opt_struct.alphas[j] - opt_struct.alphas[i])
            high = min(opt_struct.C, opt_struct.C + opt_struct.alphas[j] - opt_struct.alphas[i])
        else:
            low = max(0, opt_struct.alphas[j] + opt_struct.alphas[i] - opt_struct.C)
            high = min(opt_struct.C, opt_struct.alphas[j] + opt_struct.alphas[i])
        if low == high:
            print("low==high")
            return 0
        eta = 2.0 * opt_struct.K[i, j] - opt_struct.K[i, i] - opt_struct.K[j, j]  # 改为了用核函数的
        if eta >= 0:
            print("eta>=0")
            return 0
        opt_struct.alphas[j] -= opt_struct.labelMat[j] * (error_i - error_j) / eta
        opt_struct.alphas[j] = clip_alpha(opt_struct.alphas[j], high, low)
        update_error_k(opt_struct, j)  # 更新一下ecache
        if abs(opt_struct.alphas[j] - alpha_j_old) < 0.00001:
            print("j变化不大")
            return 0
        opt_struct.alphas[i] += opt_struct.labelMat[j] * opt_struct.labelMat[i] * (alpha_j_old - opt_struct.alphas[j])  # 对ai进行和aj一样多的改变(相反方向更新）
        update_error_k(opt_struct, i)  # 更新一下ecache
        # 为ai、aj设置一组常数项
        b1 = opt_struct.b - error_i - opt_struct.labelMat[i] * (opt_struct.alphas[i] - alpha_i_old) * opt_struct.K[i, i] - opt_struct.labelMat[j] * (opt_struct.alphas[j] - alpha_j_old) * opt_struct.K[i, j]
        b2 = opt_struct.b - error_j - opt_struct.labelMat[i] * (opt_struct.alphas[i] - alpha_i_old) * opt_struct.K[i, j] - opt_struct.labelMat[j] * (opt_struct.alphas[j] - alpha_j_old) * opt_struct.K[j, j]
        if (0 < opt_struct.alphas[i]) and (opt_struct.C > opt_struct.alphas[i]):
            opt_struct.b = b1
        elif (0 < opt_struct.alphas[j]) and (opt_struct.C > opt_struct.alphas[j]):
            opt_struct.b = b2
        else:
            opt_struct.b = (b1 + b2) / 2.0
        return 1  # 返回1，有变化
    else:
        return 0  # 返回0，没变化


def smo_platt(data_mat_in, class_labels, c, tolerate, max_iter, k_tuple=('lin', 0)):  # 初始化opt_struct的语句增加了k_tuple
    opt_struct = OptStruct(numpy.mat(data_mat_in), numpy.mat(class_labels).transpose(), c, tolerate, k_tuple)
    iteration = 0
    entire_set = True
    alpha_pairs_changed = 0
    while (iteration < max_iter) and ((alpha_pairs_changed > 0) or entire_set):
        alpha_pairs_changed = 0
        if entire_set:  # 遍历整组
            for i in range(opt_struct.m):
                alpha_pairs_changed += inner_loop(i, opt_struct)
                print("【整个数据组遍历】迭代次数为：%d，i：%d，成对改变了%d次" % (iteration, i, alpha_pairs_changed))
            iteration += 1
        else:  # 遍历非边界值
            non_bound_is = numpy.nonzero((opt_struct.alphas.A > 0) * (opt_struct.alphas.A < c))[0]  # 也就是0<alpha<c
            for i in non_bound_is:
                alpha_pairs_changed += inner_loop(i, opt_struct)
                print("【仅遍历非边界值】迭代次数为：%d，i：%d，成对改变了%d次" % (iteration, i, alpha_pairs_changed))
            iteration += 1
        if entire_set:
            entire_set = False  # toggle entire set loop
        elif alpha_pairs_changed == 0:
            entire_set = True
        print("迭代次数为：%d" % iteration)
    return opt_struct.b, opt_struct.alphas


# #####以上是有变化的函数#####
def test_rbf(k1=1.3):  # 径向基函数测试
    data_arr, label_arr = load_data_set(TrainsFile)
    b, alphas = smo_platt(data_arr, label_arr, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
    dat_mat = numpy.mat(data_arr)
    label_mat = numpy.mat(label_arr).transpose()
    sv_index = numpy.nonzero(alphas.A > 0)[0]
    select_vs = dat_mat[sv_index]  # 只要支持向量的矩阵
    label_sv = label_mat[sv_index]
    print("有%d个支持向量" % numpy.shape(select_vs)[0])
    m, n = numpy.shape(dat_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(select_vs, dat_mat[i, :], ('rbf', k1))
        predict = kernel_eval.T * numpy.multiply(label_sv, alphas[sv_index]) + b
        if numpy.sign(predict) != numpy.sign(label_arr[i]):
            error_count += 1
    print("训练组的错误率为：%f" % (float(error_count) / m))
    data_arr, label_arr = load_data_set(TestFile)
    error_count = 0
    dat_mat = numpy.mat(data_arr)
    # label_mat = numpy.mat(label_arr).transpose()
    m, n = numpy.shape(dat_mat)
    for i in range(m):
        kernel_eval = kernel_trans(select_vs, dat_mat[i, :], ('rbf', k1))
        predict = kernel_eval.T * numpy.multiply(label_sv, alphas[sv_index]) + b
        if numpy.sign(predict) != numpy.sign(label_arr[i]):
            error_count += 1
    print("测试组的错误率为：%f" % (float(error_count) / m))


# #####运行区域#####
test_rbf(k1=1.3)
