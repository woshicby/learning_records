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


def ridge_regression(x_mat, y_mat, lam=0.2):  # 岭回归
    xt_x = x_mat.T * x_mat
    denom = xt_x + numpy.eye(numpy.shape(x_mat)[1]) * lam  # 加上‘岭’
    if numpy.linalg.det(denom) == 0.0:
        print("奇异矩阵，无法求逆")
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws


def ridge_test(x_arr, y_arr):  # 岭回归测试器
    x_mat = numpy.mat(x_arr)
    y_mat = numpy.mat(y_arr).T
    y_mean = numpy.mean(y_mat, 0)  # 计算平均值，返回均值的行矩阵
    y_mat = y_mat - y_mean  # 为了消除X0，减去Y的平均值
    # 标准化x
    x_means = numpy.mean(x_mat, 0)  # 计算平均值，之后要减掉它
    x_var = numpy.var(x_mat, 0)  # 计算xi的方差，之后要除以它
    x_mat = (x_mat - x_means) / x_var
    num_test_pts = 30  # 蛮算30步
    w_mat = numpy.zeros((num_test_pts, numpy.shape(x_mat)[1]))  # 初始化一个记录每一步的系数的矩阵
    for i in range(num_test_pts):
        ws = ridge_regression(x_mat, y_mat, numpy.exp(i - 10))
        w_mat[i, :] = ws.T
    return w_mat


def draw_regression_coefficient(ridge_weights):  # 画图相关函数（8.4.1）
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_weights)
    plt.show()


# #####执行区域#####
abX, abY = load_data_set(sourceFile)
ridgeWeights = ridge_test(abX, abY)
draw_regression_coefficient(ridgeWeights)
