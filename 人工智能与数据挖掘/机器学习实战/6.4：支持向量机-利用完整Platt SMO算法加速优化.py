import numpy

# #####设置区域#####
sourceFile = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch06\testSet.txt'


# #####函数定义区域#####
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


# #####以上是已经有的函数#####
# #####以下是本节新引入的辅助函数#####
class OptStruct:  # 定义选项结构体
    def __init__(self, data_mat_in, class_labels, c, tolerate):  # 用参数来初始化结构体
        self.X = data_mat_in
        self.labelMat = class_labels
        self.C = c
        self.tol = tolerate
        self.m = numpy.shape(data_mat_in)[0]
        self.alphas = numpy.mat(numpy.zeros((self.m, 1)))
        self.b = 0
        self.eCache = numpy.mat(numpy.zeros((self.m, 2)))  # 第0维是有效标志符，误差缓存，第1维是误差


def calc_error_k(opt_struct, k):  # 求k位置的误差并返回
    f_xk = float(numpy.multiply(opt_struct.alphas, opt_struct.labelMat).T * opt_struct.X * opt_struct.X[k,:].T + opt_struct.b)
    error_k = f_xk - float(opt_struct.labelMat[k])
    return error_k


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


# #####以上是本节新引入的辅助函数#####
def inner_loop(i, opt_struct):  # 内层循环（和简化版操作差不多，就是把for i in range(m):以内的事情单独掏出来，并添加了对ecache的更新）
    error_i = calc_error_k(opt_struct, i)  # 算个ei
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
            return 0  # 返回0，没变化
        eta = 2.0 * opt_struct.X[i, :] * opt_struct.X[j, :].T - opt_struct.X[i, :] * opt_struct.X[i, :].T - opt_struct.X[j, :] * opt_struct.X[j, :].T  # aj的最优修改量
        if eta >= 0:
            print("eta>=0")
            return 0  # 返回0，没变化
        opt_struct.alphas[j] -= opt_struct.labelMat[j] * (error_i - error_j) / eta
        opt_struct.alphas[j] = clip_alpha(opt_struct.alphas[j], high, low)
        update_error_k(opt_struct, j)  # 更新一下ecache
        if abs(opt_struct.alphas[j] - alpha_j_old) < 0.00001:
            print("j变化不大")
            return 0  # 返回0，没变化
        opt_struct.alphas[i] += opt_struct.labelMat[j] * opt_struct.labelMat[i] * (alpha_j_old - opt_struct.alphas[j])  # 对ai进行和aj一样多的改变(相反方向更新）
        update_error_k(opt_struct, i)  # 更新一下ecache
        # 为ai、aj设置一组常数项
        b1 = opt_struct.b - error_i - opt_struct.labelMat[i] * (opt_struct.alphas[i] - alpha_i_old) * opt_struct.X[i, :] * opt_struct.X[i, :].T - opt_struct.labelMat[j] * (opt_struct.alphas[j] - alpha_j_old) * opt_struct.X[i, :] * opt_struct.X[i, :].T
        b2 = opt_struct.b - error_j - opt_struct.labelMat[i] * (opt_struct.alphas[i] - alpha_i_old) * opt_struct.X[i, :] * opt_struct.X[i, :].T - opt_struct.labelMat[j] * (opt_struct.alphas[j] - alpha_j_old) * opt_struct.X[i, :] * opt_struct.X[i, :].T
        if (0 < opt_struct.alphas[i]) and (opt_struct.C > opt_struct.alphas[i]):
            opt_struct.b = b1
        elif (0 < opt_struct.alphas[j]) and (opt_struct.C > opt_struct.alphas[j]):
            opt_struct.b = b2
        else:
            opt_struct.b = (b1 + b2) / 2.0
        return 1  # 返回1，有变化
    else:
        return 0  # 返回0，没变化


def smo_platt(data_mat_in, class_labels, c, tolerate, max_iter):  # 完整版Platt SMO
    opt_struct = OptStruct(numpy.mat(data_mat_in), numpy.mat(class_labels).transpose(), c, tolerate)
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


def calc_w_array(alphas, data_arr, class_labels):  # 计算超平面
    x = numpy.mat(data_arr)
    label_mat = numpy.mat(class_labels).transpose()
    m, n = numpy.shape(x)
    w = numpy.zeros((n, 1))
    for i in range(m):
        w += numpy.multiply(alphas[i] * label_mat[i], x[i, :].T)
    return w


def classify_test(alphas, data_arr, class_labels, b):#新写的测试器
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


# #####运行区域#####
dataARR, labelArr = load_data_set(sourceFile)
B, Alphas = smo_platt(dataARR, labelArr, 0.6, 0.001, 40)
print('b为%s，\n alphas为:\n' % B, Alphas)
print('alphas中非零项有', numpy.shape(Alphas[Alphas > 0])[1], '个')
classify_test(Alphas, dataARR, labelArr, B)
