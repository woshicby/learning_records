# Created on Feb 4, 2011/创建于2011年2月4日
# Translated on Feb 23, 2022/翻译于2022年2月23日
#
# Tree-Based Regression Methods/树回归方法
# @author/作者: Peter Harrington
# @translator/翻译: woshicby
# Ps.This function is modified to fit PEP 8 standard, and I have added Chinese annotations.
#    程序已经修改到符合PEP 8标准，并添加了中文注释
import numpy


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


# #####回归树相关代码开始#####
def reg_leaf(data_set):  # 生成叶节点(返回目标变量的平均值）
    return numpy.mean(data_set[:, -1])


def reg_err(data_set):  # 误差估计（返回总方差=均方差×数据集中样本的个数）
    return numpy.var(data_set[:, -1]) * numpy.shape(data_set)[0]


# #####回归树相关代码结束#####
# #####模型树相关代码开始#####
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


# #####模型树相关代码结束#####

def choose_best_split(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
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


def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):  # 假设data_set是numpy.mat所以我们可以使用数组过滤方式
    feat, val = choose_best_split(data_set, leaf_type, err_type, ops)  # 选择最好的分割方式
    if feat is None:  # 如果分割方式满足停止条件，停止并返回叶节点的值
        return val  # feat是None的时候val是leaf_type(data_set)，也就是reg_leaf(data_set)
    ret_tree = {'分割特征的序号': feat, '分割点': val}
    l_set, r_set = bin_split_data_set(data_set, feat, val)
    ret_tree['左儿子'] = create_tree(l_set, leaf_type, err_type, ops)
    ret_tree['右儿子'] = create_tree(r_set, leaf_type, err_type, ops)
    return ret_tree


# #####后剪枝开始#####
def is_tree(obj):  # 判断obj是不是树（返回布尔值）
    return type(obj).__name__ == 'dict'


def get_mean(tree):  # 对树进行塌陷处理（返回所有节点的平均值）
    if is_tree(tree['右儿子']):
        tree['右儿子'] = get_mean(tree['右儿子'])
    if is_tree(tree['左儿子']):
        tree['左儿子'] = get_mean(tree['左儿子'])
    return (tree['左儿子'] + tree['右儿子']) / 2.0


def prune(tree, test_data):  # 回归树剪枝函数
    if numpy.shape(test_data)[0] == 0:  # 没有测试数据,就对树进行塌陷处理
        return get_mean(tree)
    l_set, r_set = bin_split_data_set(test_data, tree['分割特征的序号'], tree['分割点'])
    if is_tree(tree['左儿子']):  # 如果左儿子是树，对左儿子递归剪枝
        tree['左儿子'] = prune(tree['左儿子'], l_set)
    if is_tree(tree['右儿子']):  # 如果右儿子是树，对右儿子递归剪枝
        tree['右儿子'] = prune(tree['右儿子'], r_set)
    if not is_tree(tree['左儿子']) and not is_tree(tree['右儿子']):  # 如果剪完枝之后左右儿子都是叶子，判断能否合并
        error_no_merge = sum(numpy.power(l_set[:, -1] - tree['左儿子'], 2)) + sum(numpy.power(r_set[:, -1] - tree['右儿子'], 2))  # 不合并的总方差
        tree_mean = (tree['左儿子'] + tree['右儿子']) / 2.0  # 该节点的均值
        error_merge = sum(numpy.power(test_data[:, -1] - tree_mean, 2))  # 合并的总方差
        if error_merge < error_no_merge:  # 合并后的误差更小，就合并
            print("合并左右儿子")
            return tree_mean
        else:
            return tree
    else:
        return tree


# #####后剪枝结束#####
# #####预测相关函数开始#####
def reg_tree_eval(model, in_dat):  # 为把输入和model_tree_eval()统一
    return float(model)


def model_tree_eval(model, in_dat):
    n = numpy.shape(in_dat)[1]
    x = numpy.mat(numpy.ones((1, n + 1)))
    x[:, 1:n + 1] = in_dat
    return float(x * model)


def tree_fore_cast(tree, in_data, model_eval=reg_tree_eval):  # 对任意数据，自顶向下遍历整棵树，直到命中叶节点，计算一个预测的浮点数值并返回
    if not is_tree(tree):
        return model_eval(tree, in_data)
    if in_data[tree['分割特征的序号']] > tree['分割点']:
        if is_tree(tree['左儿子']):
            return tree_fore_cast(tree['左儿子'], in_data, model_eval)
        else:
            return model_eval(tree['左儿子'], in_data)
    else:
        if is_tree(tree['右儿子']):
            return tree_fore_cast(tree['右儿子'], in_data, model_eval)
        else:
            return model_eval(tree['右儿子'], in_data)


def create_fore_cast(tree, test_data, model_eval=reg_tree_eval):  # 多次调用tree_fore_cast()，以向量形式返回一组预测值
    m = len(test_data)
    y_hat = numpy.mat(numpy.zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = tree_fore_cast(tree, numpy.mat(test_data[i]), model_eval)
    return y_hat
# #####预测相关函数结束#####
