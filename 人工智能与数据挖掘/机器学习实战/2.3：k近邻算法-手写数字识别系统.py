import os
import operator
import numpy

# #####设置区域#####
trainingDigits = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch02\digits\trainingDigits'  # 训练数据源文件路径
testDigits = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch02\digits\testDigits'  # 测试数据源文件路径


# #####函数声明区域#####
def img2vector(filename):  # 图片转向量
    return_vect = numpy.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


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


# #####执行区域#####
hand_writing_class_test()
