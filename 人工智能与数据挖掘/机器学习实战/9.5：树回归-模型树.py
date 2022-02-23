import numpy

# #####设置区域#####
sourceFile2 = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch09\ex2.txt'


# #####函数定义区域#####
def load_data_set(file_name):  # 解析\t分割的通用函数（）仅为浮点数）
    data_mat = []  # 假设最后一列是目标值
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = list(map(float, cur_line))  # map：对cur_line的每个元素使用float()
        data_mat.append(flt_line)
    return data_mat


def bin_split_data_set(data_set, feature, value):  # 通过数组过滤方式把data_set分割成两个子集并返回
    mat_0 = data_set[numpy.nonzero(data_set[:, feature] > value)[0], :]
    mat_1 = data_set[numpy.nonzero(data_set[:, feature] <= value)[0], :]
    return mat_0, mat_1


def linear_solve(data_set):  # 一个简单的线性回归
    m, n = numpy.shape(data_set)
    x = numpy.mat(numpy.ones((m, n)))
    # y = numpy.mat(numpy.ones((m, 1)))  # y只有一列
    x[:, 1:n] = data_set[:, 0:n - 1]  # 把特征值复制给x给x
    y = data_set[:, -1]  # 把结果值复制给y
    xt_x = x.T * x
    if numpy.linalg.det(xt_x) == 0.0:
        raise NameError('是奇异矩阵, 不能求逆，尝试增加ops的第二个值（数据组内条目个数阈值）')
    ws = xt_x.I * (x.T * y)  # 线性回归结果
    return ws, x, y


def model_leaf(data_set):  # 生成叶节点(创建一个线性模型并返回参数值）
    ws, x, y = linear_solve(data_set)
    return ws


def model_err(data_set):  # 误差估计（返回总方差=均方差×数据集中样本的个数）
    ws, x, y = linear_solve(data_set)
    y_hat = x * ws
    return sum(numpy.power(y - y_hat, 2))


def choose_best_split(data_set, leaf_type=model_leaf, err_type=model_err, ops=(1, 4)):
    tol_s = ops[0]  # 中方差变化量阈值
    tol_n = ops[1]  # 数据集条目数阈值
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:  # 如果所有目标变量的值都相同，则退出并返回值
        return None, leaf_type(data_set)  # 第一种退出的情况
    m, n = numpy.shape(data_set)
    # 最佳特征的选择是通过RSS误差的平均值的减少来实现的。
    s = err_type(data_set)  # s为data_set的总方差
    best_s = numpy.inf  # 初始化s的最佳值为无穷大
    best_index = 0  # 初始化最佳特征的序号为0
    best_value = 0  # 初始化最佳分割点的值为0
    for featIndex in range(n - 1):  # 对每个特征编号进行遍历
        for splitVal in set(data_set[:, featIndex].T.tolist()[0]):  # 遍历这个特征里出现过的的每一个值
            mat_0, mat_1 = bin_split_data_set(data_set, featIndex, splitVal)
            if (numpy.shape(mat_0)[0] < tol_n) or (numpy.shape(mat_1)[0] < tol_n):  # 如果切分出的左右数据集中个数小于阈值，跳出这次循环（下一个值）
                continue
            new_s = err_type(mat_0) + err_type(mat_1)  # 新的总方差
            if new_s < best_s:  # 总方差有所减小，更新
                best_index = featIndex
                best_value = splitVal
                best_s = new_s
    if (s - best_s) < tol_s:  # 如果误差减小得不大，就不要做这次分割（直接返回原数据组）
        return None, leaf_type(data_set)  # 第二种退出的情况
    mat_0, mat_1 = bin_split_data_set(data_set, best_index, best_value)  # 做分割
    if (numpy.shape(mat_0)[0] < tol_n) or (numpy.shape(mat_1)[0] < tol_n):  # 如果切分出的左右数据集中个数小于阈值，就不要做这次分割（直接返回原数据组）
        return None, leaf_type(data_set)  # 第三种退出的情况
    return best_index, best_value  # 返回最好的分割特征和分割点


def create_tree(data_set, leaf_type=model_leaf, err_type=model_err, ops=(1, 4)):  # 假设data_set是numpy.mat所以我们可以使用数组过滤方式
    feat, val = choose_best_split(data_set, leaf_type, err_type, ops)  # 选择最好的分割方式
    if feat is None:  # 如果分割方式满足停止条件，停止并返回叶节点的值
        return val  # feat是None的时候val是leaf_type(data_set)，也就是reg_leaf(data_set)
    ret_tree = {'分割特征的序号': feat, '分割点': val}
    l_set, r_set = bin_split_data_set(data_set, feat, val)
    ret_tree['左儿子'] = create_tree(l_set, leaf_type, err_type, ops)
    ret_tree['右儿子'] = create_tree(r_set, leaf_type, err_type, ops)
    return ret_tree


# #####运行区域#####
myMat2 = numpy.mat(load_data_set(sourceFile2))
print(create_tree(myMat2, model_leaf, model_err, (1, 10)))
