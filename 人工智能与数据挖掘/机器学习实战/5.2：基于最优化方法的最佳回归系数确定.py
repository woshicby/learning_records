import numpy

# #####设置区域#####
# 设置路径
sourceFilePath = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch05\testSet.txt'  # 隐形眼镜数据源文件路径


# #####函数声明区域#####
def load_data_set():  # 载入数据集
    data_mat = []
    label_mat = []
    fr = open(sourceFilePath)
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])  # 多加了一列偏正量1
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(in_x):  # sigmoid函数
    return 1.0 / (1 + numpy.exp(-in_x))


# #####梯度上升相关函数开始#####
def grad_ascent(data_mat_in, class_labels):  # 逻辑回归梯度上升（每个数据点都用）
    data_matrix = numpy.mat(data_mat_in)  # 转为numpy矩阵
    label_mat = numpy.mat(class_labels).transpose()  # 转为numpy矩阵
    m, n = numpy.shape(data_matrix)
    alpha = 0.01  # 步长
    max_cycles = 500  # 最大迭代轮次
    weights = numpy.ones((n, 1))  # 初始化回归系数向量
    for k in range(max_cycles):  # 进行max_cycles轮的循环
        h = sigmoid(data_matrix * weights)  # 对矩阵全部元素求sigmoid函数值（结果理论上应该很接近0或者1）
        error = label_mat - h  # 求与实际label的差距
        weights = weights + alpha * data_matrix.transpose() * error  # 更新回归矩阵向量值，矩阵求导的结论
    print(weights)
    return weights


def one_by_one_grad_ascent0(data_matrix, class_labels):  # 逐个梯度上升算法（stochastic gradient ascent algorith）
    m, n = numpy.shape(data_matrix)
    alpha = 0.01  # 固定不变的alpha
    weights = numpy.ones(n)  # 初始化全1
    weights_history = numpy.zeros((m, n))
    for i in range(m):  # 循环轮数为行数（每个点都学一次，遍历）
        h = sigmoid(sum(data_matrix[i] * weights))  # sigmoid(w0*x0+w1*x1+w2*x2……+wn*xn)
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]
        weights_history[i, :] = weights
    print(weights)
    return weights, weights_history


def one_by_one_grad_ascent1(data_matrix, class_labels, num_iter=500):  # 逐个梯度上升算法（stochastic gradient ascent algorith）多扫几轮
    m, n = numpy.shape(data_matrix)
    weights = numpy.ones(n)  # 初始化全1
    weights_history = numpy.zeros((num_iter * m, n))
    for j in range(num_iter):  # 学习轮数为num_iter
        for i in range(m):  # 循环轮数为行数（每个点都学一次，遍历）
            alpha = 4 / (1.0 + j + i) + 0.01  # 步长随迭代轮数减小(调参用）
            h = sigmoid(sum(data_matrix[i] * weights))  # sigmoid(w0*x0+w1*x1+w2*x2……+wn*xn)
            error = class_labels[i] - h
            weights = weights + alpha * error * data_matrix[i]
            weights_history[j * m + i, :] = weights
    print(weights)
    return weights, weights_history


def stoch_grad_ascent0(data_matrix, class_labels):  # 随机梯度上升算法（stochastic gradient ascent algorith）
    m, n = numpy.shape(data_matrix)
    alpha = 0.01  # 固定不变的alpha
    weights = numpy.ones(n)  # 初始化全1
    weights_history = numpy.zeros((m, n))
    for i in range(m):  # 循环轮数为行数（每个点都学一次，遍历）
        data_index = list(range(m))  # 重建学习名单为[0,1,2,3,……,n-1]
        rand_index = int(numpy.random.uniform(0, len(data_index)))  # 随机顺序学习
        h = sigmoid(sum(data_matrix[rand_index] * weights))
        error = class_labels[rand_index] - h
        weights = weights + alpha * error * data_matrix[rand_index]
        weights_history[i, :] = weights
    print(weights)
    return weights, weights_history


def stoch_grad_ascent1(data_matrix, class_labels, num_iter=500):  # 随机梯度上升算法（stochastic gradient ascent algorith）多扫几轮
    m, n = numpy.shape(data_matrix)
    weights = numpy.ones(n)  # 初始化全1
    weights_history = numpy.zeros((num_iter * m, n))
    for j in range(num_iter):  # 学习轮数为num_iter
        data_index = list(range(m))  # 重建学习名单为[0,1,2,3,……,n-1]
        for i in range(m):  # 每轮做一遍随机梯度上升
            alpha = 4 / (1.0 + j + i) + 0.01  # 步长随迭代轮数减小(调参用）
            rand_index = int(numpy.random.uniform(0, len(data_index)))  # 随机顺序学习
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            weights_history[j * m + i, :] = weights
            del (data_index[rand_index])  # 学过就移出学习名单
    print(weights)
    return weights, weights_history


# #####梯度上升相关函数结束#####
# #####绘图相关函数开始#####
def plot_best_fit(weights, title):  # 画出数据集和拟合直线的图像
    import matplotlib
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_data_set()  # 载入数据集
    data_arr = numpy.array(data_mat)  # 转为numpy数组（数组内元素是[1.0,特征1，特征2……特征n]）
    n = numpy.shape(data_arr)[0]  # 获取总数据条目数
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:  # 标签为1，加入cord1
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:  # 不是的话，加入cord2
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    # 开始画图
    matplotlib.rcParams['font.family'] = matplotlib.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文支持，中文字体为简体黑体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 分别画cord1和cord2的散点
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    # 画线
    x = numpy.arange(-3.0, 3.0, 0.1)  # x从-3到3，每0.1画一个点
    y = (-weights[0] - weights[1] * x) / weights[2]  # 由0=w0x0+w1x1+w2x2解出的x1与x2的关系（x是x1，y是x2）
    ax.plot(x, y)
    # 设定横纵坐标和标题
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.show()


def plot_history(my_hist):  # 画参数和迭代轮次的图像（原本是在EXTRAS\plotSDerror.py里的）
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(my_hist[:, 0])
    plt.ylabel('X0')
    ax = fig.add_subplot(312)
    ax.plot(my_hist[:, 1])
    plt.ylabel('X1')
    ax = fig.add_subplot(313)
    ax.plot(my_hist[:, 2])
    plt.xlabel('迭代次数')
    plt.ylabel('X2')
    plt.show()


# #####绘图相关函数结束#####
# #####执行区域#####
print('-----逻辑回归梯度上升-----')
dataArr, labelMat = load_data_set()
plot_best_fit(grad_ascent(dataArr, labelMat).getA(), '逻辑回归梯度上升')  # get(A)将numpy矩阵转化成数组形式
dataMat = numpy.array(dataArr)  # 把dataArr转为numpy数组
print('-----逐个遍历梯度上升-----')
weights_of_0, weightHistory_of_0 = one_by_one_grad_ascent0(dataMat, labelMat)
plot_best_fit(weights_of_0, '逐个遍历梯度上升')
plot_history(weightHistory_of_0)
print('-----逐个遍历梯度上升（多扫几轮）-----')
weights_of_0_1, weightHistory_of_0_1 = one_by_one_grad_ascent1(dataMat, labelMat, 500)
plot_best_fit(weights_of_0_1, '逐个遍历梯度上升（多扫几轮）')
plot_history(weightHistory_of_0_1)
print('-----随机遍历梯度上升-----')
weights_of_0, weightHistory_of_0 = stoch_grad_ascent0(dataMat, labelMat)
plot_best_fit(weights_of_0, '随机遍历梯度上升')
plot_history(weightHistory_of_0)
print('-----随机梯度上升（多扫几轮）-----')
weights_of_1, weightHistory_of_1 = stoch_grad_ascent1(dataMat, labelMat, 500)
plot_best_fit(weights_of_1, '逐个遍历梯度上升（多扫几轮）')
plot_history(weightHistory_of_1)
