# Created on Sep 16, 2010/创建于2010年9月16日
# Translated on Feb 11, 2022/翻译于2022年2月11日
# kNN: k Nearest Neighbors/k近邻算法
#
# Input/输入:      inX: vector to compare to existing dataset/与现有数据集进行比较的向量(1xN)
#             dataSet: size m data set of known vectors/已知向量的大小为m的数据集(NxM)
#             labels: data set labels/数据集标签(1xM vector)
#             k: number of neighbors to use for comparison(should be an odd number)/用于比较的邻居数量（应为奇数）
#
# Output/输出:     the most popular class label/最有可能的分类标签
#
# @author/作者: pbharrin
# @translator/翻译: woshicby
# Ps.This function is modified to fit PEP 8 standard, and I have added Chinese annotations.
#    程序已经修改到符合PEP 8标准，并添加了中文注释
#    Some global variables are set for easy change of file path.
#    设置了一些全局变量，方便改变文件路径
#    More output to observing the value of variables.
#    更多输出来观察变量的值
#    Provides a simple way to draw a scatter diagram.（didn't provide that about 3D, although it is also implemented in the corresponding file）
#    提供了简单的方式来画散点图（没有打包3D的画法，虽然对应文件里也实现了）
import numpy
import operator
import os
import matplotlib
import matplotlib.pyplot as plt

# #####设置区域#####
sourceFilePath = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch02\datingTestSet.txt'  # 约会例子数据源文件路径
trainingDigits = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch02\digits\trainingDigits'  # 手写数字例子训练数据源文件路径
testDigits = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch02\digits\testDigits'  # 手写数字例子测试数据源文件路径


# #####函数声明区域#####

# 该函数三个例子均有使用
def classify0(in_x, data_set, labels, k):  # k近邻分类器（对单个测试数据）
    data_set_size = data_set.shape[0]  # 若data_set为4行2列，data_set.shape[0]=4
    diff_mat = numpy.tile(in_x, (data_set_size, 1)) - data_set  # 矩阵求差
    sq_diff_mat = diff_mat ** 2  # 矩阵各项求平方
    sq_distances = sq_diff_mat.sum(axis=1)  # 平方和
    distances = sq_distances ** 0.5  # 开方得距离
    sorted_dist_indices = distances.argsort()  # 按距离大小排序，返回index
    class_count = {}  # 创建近邻字典
    # 开始抓k个近邻
    for i in range(k):
        vote_i_label = labels[sorted_dist_indices[i]]  # 取对应label
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1  # 对label对应计数+1，没有则置为零
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)  # 对class_count按第一维排序
    return sorted_class_count[0][0]  # 返回计数最多的label


# 该函数仅供简单例子使用
def create_data_set():  # 创建并返回数据集和标签
    group = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    print('训练数据集为：\n', group)
    print('对应的测试标签为：\n', labels)
    return group, labels


# 该函数仅供后两个例子使用
def auto_norm(data_set):  # 归一化
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    # norm_data_set = numpy.zeros(numpy.shape(data_set))  # 预设为全零（并没有用）
    n = data_set.shape[0]
    norm_data_set = data_set - numpy.tile(min_vals, (n, 1))  # 减去各项的最小值
    norm_data_set = norm_data_set / numpy.tile(ranges, (n, 1))  # 除以各项的极差
    return norm_data_set, ranges, min_vals


# 该函数仅供改进约会网站配对效果例子使用
def file2matrix(filename):  # 文件转矩阵
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}  # 字典型
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)  # 获取文件行数
    return_mat = numpy.zeros((number_of_lines, 3))  # 创建返回的Numpy矩阵
    class_label_vector = []  # 准备返回的标签
    index = 0  # 初始化行数为0
    for line in array_of_lines:  # 逐行添加
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]  # 前三列是三个特征值
        if list_from_line[-1].isdigit():  # isdigit() 方法检测字符串是否只由数字组成
            class_label_vector.append(int(list_from_line[-1]))
        else:
            class_label_vector.append(love_dictionary.get(list_from_line[-1]))
        index += 1  # 下一行
    return return_mat, class_label_vector


# 该函数仅供改进约会网站配对效果例子使用
def draw2d(dating_data_mat, dating_labels, i, j):  # 画二维散点图
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_data_mat[:, i], dating_data_mat[:, j], 15.0 * numpy.array(dating_labels),
               15.0 * numpy.array(dating_labels))
    matplotlib.pyplot.show()


# 该函数仅供改进约会网站配对效果例子使用
def dating_class_test():  # 分类器针对约会网站的测试代码
    ho_ratio = 0.50  # 设置测试数据占全部数据的比重
    dating_data_mat, dating_labels = file2matrix(sourceFilePath)  # 从sourceFilePath读取数据
    print('约会数据矩阵为\n', dating_data_mat, '\n约会标签矩阵为\n', dating_labels)
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)  # 归一化
    print('各项极差为', ranges, '\n各项最小值为', min_vals, '\n归一化后的数据矩阵为\n', norm_mat, )
    # draw2d(dating_data_mat, dating_labels, 0, 1)
    # draw2d(dating_data_mat, dating_labels, 1, 2)
    # draw2d(dating_data_mat, dating_labels, 0, 2)
    n = norm_mat.shape[0]  # 获取数据条目数量
    num_test_vectors = int(n * ho_ratio)  # 其中部分作为测试数据
    error_count = 0.0  # 重置错误计数
    for i in range(num_test_vectors):  # 开始测试
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vectors:n, :], dating_labels[num_test_vectors:n], 3)
        if classifier_result == dating_labels[i]:
            result = ',正确'
        else:
            error_count += 1.0
            result = ',错误,当前错误数为：' + str(error_count)
        print("分类器返回的值为:%d,正确答案为:%d" % (classifier_result, dating_labels[i]) + result)
    print("总测试数为:" + str(num_test_vectors))
    print("总错误数为:" + str(error_count))
    print("总错误率为:%f" % (error_count / float(num_test_vectors)))
    print("总错误数为:" + str(error_count))


# 该函数仅供改进约会网站配对效果例子使用
def classify_person():  # 人的分类器
    result_list = ['不喜欢的人', '魅力一般的人', '极具魅力的人']
    percent_tats = float(input("玩视频游戏所耗时间百分比为？"))
    ff_miles = float(input("每年获得的飞行常客里程数为？"))
    ice_cream = float(input("每年消费的冰淇淋有几升?"))
    dating_data_mat, dating_labels = file2matrix(sourceFilePath)
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = numpy.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat, dating_labels, 3)
    print("你可能觉得这个人是: %s" % result_list[classifier_result - 1])


# 该函数仅供手写数字识别系统例子使用
def img2vector(filename):  # 图片转向量
    return_vect = numpy.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


# 该函数仅供手写数字识别系统例子使用
def hand_writing_class_test():  # 手写分类器测试代码
    hw_labels = []  # 初始化标签列表
    training_file_list = os.listdir(trainingDigits)  # 获取训练目录的全部文件名
    m = len(training_file_list)  # 获取训练目录中有多少文件
    training_mat = numpy.zeros((m, 1024))  # 初始化训练矩阵（一行一个图像）
    for i in range(m):  # 逐个从文件名中解析出正确数字
        file_name_str = training_file_list[i]  # 获取第i个文件名
        # file_str = file_name_str.split('.')[0]  # 得到第一个.之前的内容
        class_num_int = int(file_name_str.split('_')[0])  # 得到第一个.之前的内容（转为int）
        hw_labels.append(class_num_int)  # 加入标签列表
        training_mat[i, :] = img2vector(trainingDigits + '/%s' % file_name_str)  # 加入训练矩阵
    test_file_list = os.listdir(testDigits)  # 获取测试目录的全部文件名
    error_count = 0.0  # 初始化错误计数
    m_test = len(test_file_list)  # 获取测试目录中有多少文件
    for i in range(m_test):
        # 逐个从文件名中解析出正确数字
        file_name_str = test_file_list[i]  # 获取第i个文件名
        # file_str = file_name_str.split('.')[0]  # 得到第一个.之前的内容
        class_num_int = int(file_name_str.split('_')[0])  # 得到第一个.之前的内容（转为int）
        vector_under_test = img2vector(testDigits + '/%s' % file_name_str)  # 获取一个测试向量
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)  # 进k近邻分类器
        if classifier_result == class_num_int:
            result = ',正确'
        else:
            error_count += 1.0
            result = ',错误,当前错误数为：' + str(error_count)
        print("分类器返回的值为:%d,正确答案为:%d" % (classifier_result, class_num_int) + result)
    print("\n总错误数为：%d" % error_count)
    print("\n总错误率为：%f" % (error_count / float(m_test)))
