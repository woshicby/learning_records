import numpy
import matplotlib
import matplotlib.pyplot as plt

# #####设置区域#####
TrainsFile = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch07\horseColicTraining2.txt'
TestFile = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch07\horseColicTest2.txt'


# #####函数定义区域#####
def load_data_set(file_name):  # 解析由\t分割的浮点数的通用函数
    num_feat = len(open(file_name).readline().split('\t'))  # 取特征数（实际上多了1）
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for feat_index in range(num_feat - 1):
            line_arr.append(float(cur_line[feat_index]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


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


def ada_boost_train_ds(data_arr, class_labels, num_iteration=40):  # 基于单层决策树的AdaBoost训练过程
    weak_class_arr = []
    m = numpy.shape(data_arr)[0]
    d = numpy.mat(numpy.ones((m, 1)) / m)  # 初始化各数据点的权重d，所有数值都为1/m
    agg_class_est = numpy.mat(numpy.zeros((m, 1)))  # 初始化类别估计累计值（对每个数据点）为全0
    for iteration in range(num_iteration):  # 进行num_iteration次迭代
        best_stump, error, class_est = build_stump(data_arr, class_labels, d)  # 生成单层决策树
        # print("数据点权重d为:", d.T)
        alpha = float(0.5 * numpy.log((1.0 - error) / max(error, 1e-16)))  # 计算alpha，max(error,eps) 保证不会出现除以0
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)  # 把决策树的参数存进数组里
        # print("分类结果为: ", class_est.T)
        expon = numpy.multiply(-1 * alpha * numpy.mat(class_labels).T, class_est)  # 用于计算d使用的指数, getting messy
        d = numpy.multiply(d, numpy.exp(expon))  # 为下一次迭代计算新的d
        d = d / d.sum()
        # 计算所有分类器的训练错误，如果是0次错误的话提早退出循环（用break）
        agg_class_est += alpha * class_est  # 累加上新一次的类别估计值
        # print("加权的分类结果为: ", agg_class_est.T)
        agg_errors = numpy.multiply(numpy.sign(agg_class_est) != numpy.mat(class_labels).T, numpy.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        # print("训练%i轮的错误率为:%f " % (iteration+1, error_rate))
        if error_rate == 0.0:
            break
    return weak_class_arr, agg_class_est


def plot_roc(predict_strengths, class_labels):  # 画ROC曲线用
    cur = (1.0, 1.0)  # 光标
    y_sum = 0.0  # 用于计算AUC的变量
    num_pos_clas = sum(numpy.array(class_labels) == 1.0)
    y_step = 1 / float(num_pos_clas)
    x_step = 1 / float(len(class_labels) - num_pos_clas)
    sorted_indices = predict_strengths.argsort()  # 获取排好序的索引，它是反向的
    matplotlib.rcParams['font.family'] = matplotlib.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文支持，中文字体为简体黑体
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sorted_indices.tolist()[0]:  # 遍历所有值，画一条连接每一个点的线
        if class_labels[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]
        ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')  # 画一条从cur到(cur[0]-del_x,cur[1]-del_y)的线
        cur = (cur[0] - del_x, cur[1] - del_y)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('假阳率')
    plt.ylabel('真阳率')
    plt.title('AdaBoost马疝病检测系统的ROC曲线')
    ax.axis([0, 1, 0, 1])
    print("曲线下方的面积为: ", y_sum * x_step)
    plt.show()


# #####执行区域#####
tranDatMat, tranClassLabels = load_data_set(TrainsFile)
classifierArr, aggClassEst = ada_boost_train_ds(tranDatMat, tranClassLabels, 100)
plot_roc(aggClassEst.T, tranClassLabels)