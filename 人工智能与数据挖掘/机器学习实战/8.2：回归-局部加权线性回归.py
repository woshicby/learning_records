import numpy

# #####设置区域#####
sourceFile = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch08\ex0.txt'


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
    return data_mat, label_mat  # 本章的话，输出的是xMat和yMat


def locally_weighted_linear_regression(test_point, x_arr, y_arr, k=1.0):  # 局部加权线性回归（高斯核）
    x_mat = numpy.mat(x_arr)
    y_mat = numpy.mat(y_arr).T
    m = numpy.shape(x_mat)[0]
    weights = numpy.mat(numpy.eye(m))  # 创建对角矩阵
    for j in range(m):  # 该循环创建权重矩阵（指数级衰减）
        diff_mat = test_point - x_mat[j, :]  #
        weights[j, j] = numpy.exp(diff_mat * diff_mat.T / (-2.0 * k ** 2))
    xt_x = x_mat.T * (weights * x_mat)
    if numpy.linalg.det(xt_x) == 0.0:  # 如果x转置乘x的行列式等于0
        print("是奇异矩阵, 不能求逆")
        return
    ws = xt_x.I * (x_mat.T * (weights * y_mat))
    return test_point * ws


def locally_weighted_linear_regression_test(test_arr, x_arr, y_arr, k=1.0):  # loops over all the data points and applies lwlr to each one
    m = numpy.shape(test_arr)[0]
    y_hat = numpy.zeros(m)
    for i in range(m):
        y_hat[i] = locally_weighted_linear_regression(test_arr[i], x_arr, y_arr, k)
    return y_hat


def draw_line_and_point(x_arr, y_arr, y_hat):  # 画数据点和线（8.2）
    x_mat = numpy.mat(x_arr)
    sort_index = x_mat[:, 1].argsort(0)
    x_sort = x_mat[sort_index][:, 0, :]
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = matplotlib.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文支持，中文字体为简体黑体
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_sort[:, 1], y_hat[sort_index])
    ax.scatter(x_mat[:, 1].flatten().A[0], numpy.mat(y_arr).T.flatten().A[0], s=2, c='red')
    plt.show()


# #####执行区域#####
xArr, yArr = load_data_set(sourceFile)
# print(xArr[0])
# print(locally_weighted_linear_regression(xArr[0], xArr, yArr, 1.0))
# print(locally_weighted_linear_regression(xArr[0], xArr, yArr, 0.001))
# 一下绘制不同i值的高斯核的回归结果
yHat = locally_weighted_linear_regression_test(xArr, xArr, yArr, 0.003)
draw_line_and_point(xArr, yArr, yHat)
yHat = locally_weighted_linear_regression_test(xArr, xArr, yArr, 0.01)
draw_line_and_point(xArr, yArr, yHat)
yHat = locally_weighted_linear_regression_test(xArr, xArr, yArr, 0.03)
