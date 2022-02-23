import numpy


def load_data_set(file_name):  # 解析\t分割的通用函数（）仅为浮点数）
    data_mat = []  # 假设最后一列是目标值
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = list(map(float, cur_line))  # map：对cur_line的每个元素使用float()
        data_mat.append(flt_line)
    return data_mat


def bin_split_data_set(data_set, feature, value):  # 通过数组过滤方式把data_set分割成两个子集并返回
    mat_0 = data_set[numpy.nonzero(data_set[:, feature] > value)[0], :]
    mat_1 = data_set[numpy.nonzero(data_set[:, feature] <= value)[0], :]
    return mat_0, mat_1


# #####运行区域#####
testMat = numpy.mat(numpy.eye(4))
print(testMat)
mat0, mat1 = bin_split_data_set(testMat, 1, 0.5)
print(mat0)
print(mat1)
