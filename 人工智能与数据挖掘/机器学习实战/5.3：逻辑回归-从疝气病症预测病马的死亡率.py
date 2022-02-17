import numpy

# #####设置区域#####
# 设置路径
filePath = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch05'  # 文件的存储路径
trainFileName = 'horseColicTraining.txt'  # 训练文件名字
testFileName = 'horseColicTest.txt'  # 测试文件名字


# #####函数声明区域#####
def sigmoid(in_x):  # sigmoid函数
    return 1.0 / (1 + numpy.exp(-in_x))


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
    print(weights.flatten())
    return weights, weights_history


def classify_vector(in_x, weights):  # 根据prob值输出结果
    prob = sigmoid(sum(in_x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():  # 训练和测试器
    fr_train = open(filePath + '/%s' % trainFileName)
    fr_test = open(filePath + '/%s' % testFileName)
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    train_weights = stoch_grad_ascent1(numpy.array(training_set), training_labels, 1000)[0]
    error_count = 0
    num_test_vec = 0.0
    for line in fr_test.readlines():
        num_test_vec += 1.0
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        if int(classify_vector(numpy.array(line_arr), train_weights)) != int(curr_line[21]):
            error_count += 1
    error_rate = (float(error_count) / num_test_vec)
    print("错误率为:%f" % error_rate)
    return error_rate


def multi_test():  # 进行多轮训练和测试求平均错误率
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print("%d次迭代之后的平均错误率为%f" % (num_tests, error_sum / float(num_tests)))


# #####执行区域#####
multi_test()
