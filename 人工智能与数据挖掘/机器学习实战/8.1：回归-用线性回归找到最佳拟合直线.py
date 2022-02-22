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


def standard_regression(x_arr, y_arr):  # 线性回归
    x_mat = numpy.mat(x_arr)
    y_mat = numpy.mat(y_arr).T
    xt_x = x_mat.T * x_mat
    if numpy.linalg.det(xt_x) == 0.0:  # 如果x转置乘x的行列式等于0
        print("是奇异矩阵, 不能求逆")
        return
    ws = xt_x.I * (x_mat.T * y_mat)
    return ws


def draw_line_and_point(x_mat, y_mat, wire_stand):  # 画数据点和线（8.1）
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = matplotlib.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文支持，中文字体为简体黑体
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])
    x_copy = x_mat.copy()
    x_copy.sort(0)
    y_hat = x_copy * wire_stand
    ax.plot(x_copy[:, 1], y_hat)
    plt.show()


# #####执行区域#####
xArr, yArr = load_data_set(sourceFile)
# print(xArr[0:2])
wireStand = standard_regression(xArr, yArr)
# print(wireStand)
xMat = numpy.mat(xArr)
yMat = numpy.mat(yArr)
yHat = xMat * wireStand
draw_line_and_point(xMat, yMat, wireStand)
print(numpy.corrcoef(yHat.T, yMat))  # 用于计算相关系数（两个参数都是行向量）
