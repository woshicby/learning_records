import numpy


def load_simp_data():  # 载入简单例子
    dat_mat = numpy.matrix([[1., 2.1],
                            [2., 1.1],
                            [1.3, 1.],
                            [1., 1.],
                            [2., 1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dat_mat, class_labels


def stump_classify(data_matrix, dimension, thresh_val, thresh_ineq):  # 仅仅是分类数据
    ret_array = numpy.ones((numpy.shape(data_matrix)[0], 1))
    if thresh_ineq == '小于':
        ret_array[data_matrix[:, dimension] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimension] > thresh_val] = -1.0
    return ret_array


def build_stump(data_arr, class_labels, d):  # 生成单层决策树
    data_matrix = numpy.mat(data_arr)  # 转为numpy矩阵
    label_matrix = numpy.mat(class_labels).T  # 转为numpy矩阵并转置
    m, n = numpy.shape(data_matrix)  # 获取特征数和条目数
    num_steps = 10.0  # 用于在特征的所有可能值上进行遍历
    best_stump = {}  # 用于存储给定权重向量d时所得到的最佳单层决策树
    best_clas_est = numpy.mat(numpy.zeros((m, 1)))
    min_error = numpy.inf  # 用于记录最小错误率（初始化为无穷大）
    for feat_index in range(n):  # 对每一个特征循环
        range_min = data_matrix[:, feat_index].min()  # 最小值
        range_max = data_matrix[:, feat_index].max()  # 最大值
        step_size = (range_max - range_min) / num_steps  # 求步长
        for j in range(-1, int(num_steps) + 1):  # 对每个步长循环
            for unequal_sign in ['小于', '大于']:  # 对每个不等号循环（切换大于小于）
                thresh_val = (range_min + float(j) * step_size)  # 求给stump_class()用的阈值
                predicted_vals = stump_classify(data_matrix, feat_index, thresh_val, unequal_sign)  # 调用stump_class()
                err_vector = numpy.mat(numpy.ones((m, 1)))  # 初始化error矩阵为全1
                err_vector[predicted_vals == label_matrix] = 0  # 预测值与标签相同的置为0
                weighted_error = d.T * err_vector  # 乘以权重d计算加权错误率
                # print("目前维数：%d，阈值：%.2f，不等号：%s，加权错误率为：%.3f" % (feat_index + 1, thresh_val, unequal_sign, weighted_error))
                if weighted_error < min_error:  # 若加权错误率小于记录的最小值，更新相关数据
                    min_error = weighted_error
                    best_clas_est = predicted_vals.copy()
                    best_stump['维数'] = feat_index
                    best_stump['阈值'] = thresh_val
                    best_stump['不等号'] = unequal_sign
                    # print('【更新最小值】当前最佳维数为%d，最佳阈值为%.2f，不等号为：%s' % (feat_index + 1, thresh_val, unequal_sign))
    return best_stump, min_error, best_clas_est


# #####执行区域#####
datMat, classLabels = load_simp_data()
D = numpy.mat(numpy.ones((5, 1)) / 5)
build_stump(datMat, classLabels, D)
