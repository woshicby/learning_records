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


def standard_regression(x_arr, y_arr):  # 线性回归
    x_mat = numpy.mat(x_arr)
    y_mat = numpy.mat(y_arr).T
    xt_x = x_mat.T * x_mat
    if numpy.linalg.det(xt_x) == 0.0:
        print("是奇异矩阵, 不能求逆")
        return
    ws = xt_x.I * (x_mat.T * y_mat)
    return ws


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


def locally_weighted_linear_regression_test(test_arr, x_arr, y_arr, k=1.0):  # 遍历所有测试点并计算局部线性回归的结果
    m = numpy.shape(test_arr)[0]
    y_hat = numpy.zeros(m)
    for i in range(m):
        y_hat[i] = locally_weighted_linear_regression(test_arr[i], x_arr, y_arr, k)
    return y_hat


# #####执行区域#####
abX, abY = load_data_set(sourceFile)
print('-----训练-----')
yHat01 = locally_weighted_linear_regression_test(abX[0:99], abX[0:99], abY[0:99], 0.1)
print('高斯核的k=0.1时对训练集的误差为：', rss_error(abY[0:99], yHat01.T))
yHat1 = locally_weighted_linear_regression_test(abX[0:99], abX[0:99], abY[0:99], 1)
print('高斯核的k=1时对训练集的误差为：', rss_error(abY[0:99], yHat1.T))
yHat10 = locally_weighted_linear_regression_test(abX[0:99], abX[0:99], abY[0:99], 10)
print('高斯核的k=10时对训练集的误差为：', rss_error(abY[0:99], yHat10.T))
print('-----测试-----')
yHat1 = locally_weighted_linear_regression_test(abX[0:99], abX[0:99], abY[0:99], 0.1)
print('高斯核的k=0.1时对测试集的误差为：', rss_error(abY[100:199], yHat01.T))
yHat1 = locally_weighted_linear_regression_test(abX[100:199], abX[0:99], abY[0:99], 1)
print('高斯核的k=1时对测试集的误差为：', rss_error(abY[100:199], yHat1.T))
yHat10 = locally_weighted_linear_regression_test(abX[100:199], abX[0:99], abY[0:99], 10)
print('高斯核的k=10时对测试集的误差为：', rss_error(abY[100:199], yHat10.T))
print('-----标准线性回归-----')
wareStandard = standard_regression(abX[0:99], abY[0:99])
yHat = numpy.mat(abX[100:199]) * wareStandard
print('标准线性回归的误差为', rss_error(abY[100:199], yHat.T.A))
