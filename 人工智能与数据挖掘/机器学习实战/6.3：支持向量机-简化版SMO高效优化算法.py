import numpy

# #####设置区域#####
sourceFile = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch06\testSet.txt'


# #####函数定义区域#####
# #####辅助函数#####
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


# #####辅助函数结束#####
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
                print("连续无更新次数:%d，i:%d，成对改变了%d次" % (iteration, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:  # 若本次没更新，计数+1
            iteration += 1
        else:  # 有更新，回归0
            iteration = 0
        print("连续无更新次数: %d" % iteration)
    return b, alphas


# #####运行区域#####
dataARR, labelArr = load_data_set(sourceFile)
B, Alphas = smo_simple(dataARR, labelArr, 0.6, 0.001, 40)
print('b为%s，\n alphas为:\n' % B, Alphas)
print('alphas中非零项有', numpy.shape(Alphas[Alphas > 0]), '个')
