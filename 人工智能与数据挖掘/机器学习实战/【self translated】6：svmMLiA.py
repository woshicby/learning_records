# Created on Nov 4, 2010/创建于2010年11月4日
# Translated on Feb 18, 2022/翻译于2022年2月18日
#
# Chapter 5 source file for Machine Learning in Action
# @author/作者:Peter
# @translator/翻译: woshicby
# Ps.This function is modified to fit PEP 8 standard, and I have added Chinese annotations.
#    程序已经修改到符合PEP 8标准，并添加了中文注释
import numpy

# #####设置区域#####
TrainsFile = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch06\testSetRBF.txt'  # 训练文件路径（6.5）
TestFile = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch06\testSetRBF2.txt'  # 测试文件路径（6.5）
trainingDigits = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch02\digits\trainingDigits'  # 训练数据源文件路径（6.6）
testDigits = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch02\digits\testDigits'  # 测试数据源文件路径（6.6）


# #####函数定义区域#####
# #####以下是6.3简化版SMO算法相关函数#####
def load_data_set(file_name):  # 读入数据组
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def select_j_rand(i, m):  # i：第一个alpha的下标；m：所有alpha的数目
    j = i  # 要选一个和i不一样的j
    while j == i:
        j = int(numpy.random.uniform(0, m))  # 相等就重新抽
    return j


def clip_alpha(aj, high, low):  # 大于high就把aj调成high，小于low就把aj调成low
    if aj > high:
        aj = high
    if low > aj:
        aj = low
    return aj


def smo_simple(data_mat_in, class_labels, c, tolerate, max_iter):  # 简化版SMO算法
    data_matrix = numpy.mat(data_mat_in)  # 转为numpy矩阵
    label_mat = numpy.mat(class_labels).transpose()  # 转为numpy矩阵并转置（变为行向量，类型为numpy.matrix）
    b = 0  # 常数项初始化为0
    m, n = numpy.shape(data_matrix)  # 取data_matrix的形状
    alphas = numpy.mat(numpy.zeros((m, 1)))  # 初始化alpha为0向量
    iteration = 0  # 迭代次数初始化为0
    while iteration < max_iter:  # 连续max_iter轮无更新，就迭代结束
        alpha_pairs_changed = 0  # alpha成对改变技术初始化为0
        for i in range(m):  # 扫一遍数据组
            f_xi = float(numpy.multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b  # f（xi）=预测结果
            error_i = f_xi - float(label_mat[i])  # if语句检查样本是否违反KKT条件
            # tolerate：容忍错误的程度
            # labelMat[i]*Ei < -tolerate，且不>=C，则alphas[i]要增大
            # labelMat[i]*Ei > tolerate，且不<=0,则alphas[i]要减小
            if ((label_mat[i] * error_i < -tolerate) and (alphas[i] < c)) or ((label_mat[i] * error_i > tolerate) and (alphas[i] > 0)):
                j = select_j_rand(i, m)
                f_xj = float(numpy.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                error_j = f_xj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()  # 保存原ai
                alpha_j_old = alphas[j].copy()  # 保存原aj
                if label_mat[i] != label_mat[j]:  # i、j不同类
                    low = max(0, alphas[j] - alphas[i])
                    high = min(c, c + alphas[j] - alphas[i])
                else:  # i、j同类
                    low = max(0, alphas[j] + alphas[i] - c)
                    high = min(c, alphas[j] + alphas[i])
                if low == high:
                    print("low==high")
                    continue  # low==high，没法调，过。
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T - data_matrix[j, :] * data_matrix[j, :].T  # aj的最优修改量
                if eta >= 0:
                    print("eta>=0")
                    continue  # 数据集间隔大于零，过。（简化了）
                alphas[j] -= label_mat[j] * (error_i - error_j) / eta
                alphas[j] = clip_alpha(alphas[j], high, low)  # 把aj调整到0和c之间
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print("j动得不多")
                    continue  # j动得不多，过。
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])  # 对ai进行和aj一样多的改变(相反方向更新）
                # 为ai、aj设置一组常数项
                b1 = b - error_i - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - error_j - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T - label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                if (0 < alphas[i]) and (c > alphas[i]):  # 0<ai<c，b取b1
                    b = b1
                elif (0 < alphas[j]) and (c > alphas[j]):  # 0<aj<c，b取b2
                    b = b2
                else:  # 两个都不在0和c之间，b取b1、b2的平均值
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print("连续无更新次数：%d，i：%d，成对改变了%d次" % (iteration, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:  # 若本次没更新，计数+1
            iteration += 1
        else:  # 有更新，回归0
            iteration = 0
        print("连续无更新次数: %d" % iteration)
    return b, alphas


# #####以上是6.3简化版SMO算法相关函数#####
# #####以下是6.4完整版Platt SMO算法相关函数#####
# class OptStruct:  # 定义选项结构体
#     def __init__(self, data_mat_in, class_labels, c, tolerate):  # 用参数来初始化结构体
#         self.X = data_mat_in
#         self.labelMat = class_labels
#         self.C = c
#         self.tol = tolerate
#         self.m = numpy.shape(data_mat_in)[0]
#         self.alphas = numpy.mat(numpy.zeros((self.m, 1)))
#         self.b = 0
#         self.eCache = numpy.mat(numpy.zeros((self.m, 2)))  # 第0维是有效标志符，误差缓存，第1维是误差
#
#
# def calc_error_k(opt_struct, k):  # 求k位置的误差并返回
#     f_xk = float(numpy.multiply(opt_struct.alphas, opt_struct.labelMat).T * opt_struct.X[:, k] * opt_struct.X[:, k].T + opt_struct.b)
#     error_k = f_xk - float(opt_struct.labelMat[k])
#     return error_k
#
#
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


# def inner_loop(i, opt_struct):  # 内层循环（和简化版操作差不多，就是把for i in range(m):以内的事情单独掏出来，并添加了对ecache的更新）
#     error_i = calc_error_k(opt_struct, i)  # 算个ei
#     # opt_struct.tol：容忍错误的程度
#     # labelMat[i]*Ei < -opt_struct.tol，且不>=C，则alphas[i]要增大
#     # labelMat[i]*Ei > opt_struct.tol，且不<=0,则alphas[i]要减小
#     if ((opt_struct.labelMat[i] * error_i < -opt_struct.tol) and (opt_struct.alphas[i] < opt_struct.C)) or ((opt_struct.labelMat[i] * error_i > opt_struct.tol) and (opt_struct.alphas[i] > 0)):
#         j, error_j = select_j(i, opt_struct, error_i)  # 第一次选择，结果是用select_j_rand产生的
#         alpha_i_old = opt_struct.alphas[i].copy()
#         alpha_j_old = opt_struct.alphas[j].copy()
#         if opt_struct.labelMat[i] != opt_struct.labelMat[j]:
#             low = max(0, opt_struct.alphas[j] - opt_struct.alphas[i])
#             high = min(opt_struct.C, opt_struct.C + opt_struct.alphas[j] - opt_struct.alphas[i])
#         else:
#             low = max(0, opt_struct.alphas[j] + opt_struct.alphas[i] - opt_struct.C)
#             high = min(opt_struct.C, opt_struct.alphas[j] + opt_struct.alphas[i])
#         if low == high:
#             print("low==high")
#             return 0  # 返回0，没变化
#         eta = 2.0 * opt_struct.X[i, :] * opt_struct.X[j, :].T - opt_struct.X[i, :] * opt_struct.X[i, :].T - opt_struct.X[j, :] * opt_struct.X[j, :].T  # aj的最优修改量
#         if eta >= 0:
#             print("eta>=0")
#             return 0  # 返回0，没变化
#         opt_struct.alphas[j] -= opt_struct.labelMat[j] * (error_i - error_j) / eta
#         opt_struct.alphas[j] = clip_alpha(opt_struct.alphas[j], high, low)
#         update_error_k(opt_struct, j)  # 更新一下ecache
#         if abs(opt_struct.alphas[j] - alpha_j_old) < 0.00001:
#             print("j变化不大")
#             return 0  # 返回0，没变化
#         opt_struct.alphas[i] += opt_struct.labelMat[j] * opt_struct.labelMat[i] * (alpha_j_old - opt_struct.alphas[j])  # 对ai进行和aj一样多的改变(相反方向更新）
#         update_error_k(opt_struct, i)  # 更新一下ecache
#         # 为ai、aj设置一组常数项
#         b1 = opt_struct.b - error_i - opt_struct.labelMat[i] * (opt_struct.alphas[i] - alpha_i_old) * opt_struct.X[i, :] * opt_struct.X[i, :].T - opt_struct.labelMat[j] * (opt_struct.alphas[j] - alpha_j_old) * opt_struct.X[i, :] * opt_struct.X[i, :].T
#         b2 = opt_struct.b - error_j - opt_struct.labelMat[i] * (opt_struct.alphas[i] - alpha_i_old) * opt_struct.X[i, :] * opt_struct.X[i, :].T - opt_struct.labelMat[j] * (opt_struct.alphas[j] - alpha_j_old) * opt_struct.X[i, :] * opt_struct.X[i, :].T
#         if (0 < opt_struct.alphas[i]) and (opt_struct.C > opt_struct.alphas[i]):
#             opt_struct.b = b1
#         elif (0 < opt_struct.alphas[j]) and (opt_struct.C > opt_struct.alphas[j]):
#             opt_struct.b = b2
#         else:
#             opt_struct.b = (b1 + b2) / 2.0
#         return 1  # 返回1，有变化
#     else:
#         return 0  # 返回0，没变化
#
#
# def smo_platt(data_mat_in, class_labels, c, tolerate, max_iter):  # 完整版Platt SMO
#     opt_struct = OptStruct(numpy.mat(data_mat_in), numpy.mat(class_labels).transpose(), c, tolerate)
#     iteration = 0
#     entire_set = True
#     alpha_pairs_changed = 0
#     while (iteration < max_iter) and ((alpha_pairs_changed > 0) or entire_set):
#         alpha_pairs_changed = 0
#         if entire_set:  # 遍历整组
#             for i in range(opt_struct.m):
#                 alpha_pairs_changed += inner_loop(i, opt_struct)
#                 print("【整个数据组遍历】迭代次数为：%d，i：%d，成对改变了%d次" % (iteration, i, alpha_pairs_changed))
#             iteration += 1
#         else:  # 遍历非边界值
#             non_bound_is = numpy.nonzero((opt_struct.alphas.A > 0) * (opt_struct.alphas.A < c))[0]  # 也就是0<alpha<c
#             for i in non_bound_is:
#                 alpha_pairs_changed += inner_loop(i, opt_struct)
#                 print("【仅遍历非边界值】迭代次数为：%d，i：%d，成对改变了%d次" % (iteration, i, alpha_pairs_changed))
#             iteration += 1
#         if entire_set:
#             entire_set = False  # toggle entire set loop
#         elif alpha_pairs_changed == 0:
#             entire_set = True
#         print("迭代次数为：%d" % iteration)
#     return opt_struct.b, opt_struct.alphas
#
#
# #####以上是6.4完整版Platt SMO算法相关函数#####
def calc_w_array(alphas, data_arr, class_labels):  # 计算超平面
    x = numpy.mat(data_arr)
    label_mat = numpy.mat(class_labels).transpose()
    m, n = numpy.shape(x)
    w = numpy.zeros((n, 1))
    for i in range(m):
        w += numpy.multiply(alphas[i] * label_mat[i], x[i, :].T)
    return w


def classify_test(alphas, data_arr, class_labels, b):  # 6.4用的测试器（这个原本的文件里没有写）
    dat_mat = numpy.mat(data_arr)
    results = [(dat_mat[i] * numpy.mat(calc_w_array(alphas, data_arr, class_labels)) + b) for i in range(numpy.shape(dat_mat)[0])]
    error_count = 0
    for i in range(len(results)):
        if results[i] > 0:
            results[i] = 1
        else:
            results[i] = -1
        if results[i] == class_labels[i]:
            print('第%i个数据分类无误，当前错误率为%f' % (i + 1, float(error_count / (i + 1))))
        else:
            print('第%i个数据分类错误，当前错误率为%f' % (i + 1, float(error_count / (i + 1))))


# #####以下是6.5加入核函数的完整版Platt SMO算法相关函数#####
def kernel_trans(x, a, k_tuple):  # 计算核函数/把数据映射到高维空间
    m, n = numpy.shape(x)
    k = numpy.mat(numpy.zeros((m, 1)))
    if k_tuple[0] == 'lin':
        k = x * a.T  # 线性核函数
    elif k_tuple[0] == 'rbf':
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


# select_j()和update_error_k（）没有变化，沿用前面的版本
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


# #####以上是6.5加入核函数的完整版Platt SMO算法相关函数#####
# #####以下是只有6.6手写识别问题回顾用到的函数#####
def img2vector(filename):  # 图片转向量（和第二章那个一毛一样）
    return_vect = numpy.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def load_images(dir_name):
    import os
    hw_labels = []  # 初始化标签列表
    training_file_list = os.listdir(dir_name)  # 获取训练目录的全部文件名
    m = len(training_file_list)  # 获取训练目录中有多少文件
    training_mat = numpy.zeros((m, 1024))  # 初始化训练矩阵（一行一个图像）
    for i in range(m):  # 逐个从文件名中解析出正确数字
        file_name_str = training_file_list[i]  # 获取第i个文件名
        # fileStr = file_name_str.split('.')[0]  # 得到第一个.之前的内容
        class_num_int = int(file_name_str.split('_')[0])  # 得到第一个.之前的内容（转为int）
        if class_num_int == 9:
            hw_labels.append(-1)
        else:
            hw_labels.append(1)
        training_mat[i, :] = img2vector('%s/%s' % (dir_name, file_name_str))
    return training_mat, hw_labels


def test_digits(k_tuple=('rbf', 10)):
    data_arr, label_arr = load_images(trainingDigits)
    b, alphas = smo_platt(data_arr, label_arr, 200, 0.0001, 10000, k_tuple)
    dat_mat = numpy.mat(data_arr)
    label_mat = numpy.mat(label_arr).transpose()
    sv_index = numpy.nonzero(alphas.A > 0)[0]
    select_vs = dat_mat[sv_index]
    label_sv = label_mat[sv_index]
    print("有%d个支持向量" % numpy.shape(select_vs)[0])
    m, n = numpy.shape(dat_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(select_vs, dat_mat[i, :], k_tuple)
        predict = kernel_eval.T * numpy.multiply(label_sv, alphas[sv_index]) + b
        if numpy.sign(predict) != numpy.sign(label_arr[i]):
            error_count += 1
    print("训练组的错误率为：%f" % (float(error_count) / m))
    data_arr, label_arr = load_images(testDigits)
    error_count = 0
    dat_mat = numpy.mat(data_arr)
    # label_mat = numpy.mat(label_arr).transpose()
    m, n = numpy.shape(dat_mat)
    for i in range(m):
        kernel_eval = kernel_trans(select_vs, dat_mat[i, :], k_tuple)
        predict = kernel_eval.T * numpy.multiply(label_sv, alphas[sv_index]) + b
        if numpy.sign(predict) != numpy.sign(label_arr[i]):
            error_count += 1
    print("测试组的错误率为：%f" % (float(error_count) / m))
