import numpy

# #####设置区域#####
sourceFile = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch08\abalone.txt'


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


def rss_error(y_arr, y_hat_arr):  # 返回一个误差值
    return ((y_arr - y_hat_arr) ** 2).sum()


def regularize(x_mat):  # 逐列标准化
    in_mat = x_mat.copy()
    in_means = numpy.mean(in_mat, 0)  # 计算平均值，之后要减掉它
    in_var = numpy.var(in_mat, 0)  # 计算xi的方差，之后要除以它
    in_mat = (in_mat - in_means) / in_var
    return in_mat


def stage_wise(x_arr, y_arr, eps=0.01, num_it=100):
    x_mat = numpy.mat(x_arr)
    y_mat = numpy.mat(y_arr).T
    y_mean = numpy.mean(y_mat, 0)
    y_mat = y_mat - y_mean  # 也能标准化ys，但是会得到更小的coef
    x_mat = regularize(x_mat)
    m, n = numpy.shape(x_mat)
    return_mat = numpy.zeros((num_it, n))  # 初始化一个记录每一代系数的矩阵
    ws = numpy.zeros((n, 1))  # ws记录当代的系数
    # ws_test = ws.copy()
    ws_max = ws.copy()  # 初始化一个存储最大值的列向量
    for i in range(num_it):  # 开始迭代num_it次
        # print(ws.T)
        lowest_error = numpy.inf  # 初始化最小错误值为无穷大
        for j in range(n):  # 遍历每个特征
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps * sign
                y_test = x_mat * ws_test
                rss_e = rss_error(y_mat.A, y_test.A)
                if rss_e < lowest_error:  # 如果错误更小的话就更新
                    lowest_error = rss_e
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T
    return return_mat, ws


def standard_regression(x_arr, y_arr):  # 线性回归
    x_mat = numpy.mat(x_arr)
    y_mat = numpy.mat(y_arr).T
    xt_x = x_mat.T * x_mat
    if numpy.linalg.det(xt_x) == 0.0:
        print("是奇异矩阵, 不能求逆")
        return
    ws = xt_x.I * (x_mat.T * y_mat)
    return ws


def least_square_method(x_arr, y_arr):  # 最小二乘法
    x_mat = numpy.mat(x_arr)
    y_mat = numpy.mat(y_arr).T
    x_mat = regularize(x_mat)
    y_mean = numpy.mean(y_mat, 0)
    y_mat -= y_mean
    weights = standard_regression(x_mat, y_mat.T)
    return weights.T


def draw_regression_coefficient(ridge_weights):  # 画图相关函数（8.4.1）
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_weights)
    plt.show()


# #####执行区域#####
xArr, yArr = load_data_set(sourceFile)
print('-----步长0.01，迭代200回-----')
print(stage_wise(xArr, yArr, 0.01, 200)[1])
draw_regression_coefficient(stage_wise(xArr, yArr, 0.01, 200)[0])
print('-----步长0.001，迭代5000回-----')
print(stage_wise(xArr, yArr, 0.001, 5000)[1])
draw_regression_coefficient(stage_wise(xArr, yArr, 0.001, 5000)[0])
print('-----最小二乘法-----')
print(least_square_method(xArr, yArr))
