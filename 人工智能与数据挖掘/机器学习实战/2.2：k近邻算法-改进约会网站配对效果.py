import numpy
import operator
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
# #####设置区域#####
sourceFilePath = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch02\datingTestSet.txt'  # 训练数据源文件路径


# #####函数声明区域#####
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


def auto_norm(data_set):  # 归一化
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    # norm_data_set = numpy.zeros(numpy.shape(data_set))  # 预设为全零（并没有用）
    n = data_set.shape[0]
    norm_data_set = data_set - numpy.tile(min_vals, (n, 1))  # 减去各项的最小值
    norm_data_set = norm_data_set / numpy.tile(ranges, (n, 1))  # 除以各项的极差
    return norm_data_set, ranges, min_vals


def draw2d(dating_data_mat, dating_labels, i, j):
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_data_mat[:, i], dating_data_mat[:, j], 15.0 * numpy.array(dating_labels),
               15.0 * numpy.array(dating_labels))
    matplotlib.pyplot.show()


def draw3d(dating_data_mat, dating_labels):
    # 绘制散点图
    matplotlib.rcParams['font.family'] = matplotlib.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文支持，中文字体为简体黑体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = matplotlib.pyplot.figure()
    ax = p3d.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1], dating_data_mat[:, 2], 15.0 * numpy.array(dating_labels), 15.0 * numpy.array(dating_labels), 15.0 * numpy.array(dating_labels))
    '''
    # 绘制图例
    ax.legend(loc='best')
    '''
    # 添加坐标轴
    ax.set_xlabel('每年获得的飞行常客里程数', fontdict={'size': 10, 'color': 'red'})
    ax.set_ylabel('玩视频游戏所耗时间百分比', fontdict={'size': 10, 'color': 'red'})
    ax.set_zlabel('每年所消费的冰淇淋公升数', fontdict={'size': 10, 'color': 'red'})
    ax.set_title('三维点云图')
    matplotlib.pyplot.show()


def dating_class_test():  # 分类器针对约会网站的测试代码
    ho_ratio = 0.50  # 设置测试数据占全部数据的比重
    dating_data_mat, dating_labels = file2matrix(sourceFilePath)  # 从sourceFilePath读取数据
    print('约会数据矩阵为\n', dating_data_mat, '\n约会标签矩阵为\n', dating_labels)
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)  # 归一化
    print('各项极差为', ranges, '\n各项最小值为', min_vals, '\n归一化后的数据矩阵为\n', norm_mat, )
    # draw3d(dating_data_mat, dating_labels)
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


# #####执行区域#####
dating_class_test()
# classify_person()
