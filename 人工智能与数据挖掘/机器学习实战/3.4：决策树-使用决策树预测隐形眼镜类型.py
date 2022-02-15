import math
import operator
import matplotlib
import matplotlib.pyplot as plt
# #####设置区域#####
# 设置路径
filePath = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch03'  # 生成树的存储路径
fileName = 'MyTree.txt'  # 存储生成树的文件名
sourceFilePath = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch03\lenses.txt'  # 隐形眼镜数据源文件路径
# 定义文本框和箭头格式
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# #####函数声明区域#####
def create_data_set():  # 生成简单的鱼鉴定数据集
    data_set = [[1, 1, '是'],
                [1, 1, '是'],
                [1, 0, '不是'],
                [0, 1, '不是'],
                [0, 1, '不是']]
    labels = ['不浮出水面', '有脚蹼']
    # 更改为离散值
    return data_set, labels


# #####以下是生成决策树相关函数（来自3.1：决策树的构造.py）#####
def calc_shannon_ent(data_set):  # 求香农熵/信息熵
    num_entries = len(data_set)  # 计算实例总数
    label_counts = {}  # 标签计数字典
    for featVec in data_set:  # 生成标签计数字典的数值
        current_label = featVec[-1]  # 提取当前标签
        if current_label not in label_counts.keys():  # 若标签计数字典中没记录
            label_counts[current_label] = 0  # 则初始化为0
        label_counts[current_label] += 1  # 进行一个计数的加1
    shannon_ent = 0.0  # 香农熵初始化为0
    for key in label_counts:  # 遍历标签计数字典
        prob = float(label_counts[key]) / num_entries  # 求选择该标签的概率
        shannon_ent -= prob * math.log(prob, 2)  # 累减prob*log2prob
    return shannon_ent


def calc_gini_impurity(data_set):  # 求基尼不纯度
    num_entries = len(data_set)  # 计算实例总数
    label_counts = {}  # 标签计数字典
    for featVec in data_set:  # 生成标签计数字典的数值
        current_label = featVec[-1]  # 提取当前标签
        if current_label not in label_counts.keys():  # 若标签计数字典中没记录
            label_counts[current_label] = 0  # 则初始化为0
        label_counts[current_label] += 1  # 进行一个计数的加1
    gini_impurity = 1.0  # 基尼不纯度初始化为1
    for key in label_counts:  # 遍历标签计数字典
        prob = float(label_counts[key]) / num_entries  # 求选择该标签的概率
        gini_impurity -= prob ** 2  # 累减prob**2
    return gini_impurity


def split_data_set(data_set, axis, value):  # 划分数据集
    ret_data_set = []  # 创建新列表
    for featVec in data_set:  # 逐个检查
        if featVec[axis] == value:  # 若featVec的第axis维符合预设的value值
            reduced_feat_vec = featVec[:axis]  # 提取axis前的那些特征
            reduced_feat_vec.extend(featVec[axis + 1:])  # 加入axis后的那些特征
            ret_data_set.append(reduced_feat_vec)  # 把该项加入列表
    return ret_data_set  # 返回列表


def choose_best_feature_to_split(data_set):  # 选择最好的划分方式(按照香农熵）
    num_features = len(data_set[0]) - 1  # 特征数等于每一行的元素数-1（因为最后一个是标签）
    base_entropy = calc_shannon_ent(data_set)  # 求香浓熵的原始值
    best_info_gain = 0.0  # 初始化最好的信息增益为0.0
    best_feature = -1  # 初始化最好的分割axis为-1
    for i in range(num_features):  # 遍历各特征
        feat_list = [example[i] for example in data_set]  # 创建一个这个特征的所有值的列表
        unique_vals = set(feat_list)  # 转换为集合
        new_entropy = 0.0  # 新的香浓熵初始化为0.0
        for value in unique_vals:  # 遍历特征的值的集合
            sub_data_set = split_data_set(data_set, i, value)  # 尝试划分
            prob = len(sub_data_set) / float(len(data_set))  # 求该划分的占比
            # print('第', i, '维内容为', value, '的该划分的占比为：', prob, '基尼不纯度为：', calc_shannon_ent(sub_data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)  # 累加这个划分的占比*这个划分的香农熵
        # print('以第', i, '维划分的香农熵为：', new_entropy)
        info_gain = base_entropy - new_entropy  # 计算信息增益（也就是香农熵的差值）
        # print('以第', i, '维划分的信息增益为：', info_gain)
        if info_gain > best_info_gain:  # 把正在计算的这个和目前最好的相比较
            best_info_gain = info_gain  # 更好的就替换
            best_feature = i  # 最好地划分特征也替换为当前特征
    return best_feature  # 返回最好地划分特征对应的维数


def choose_best_feature_to_split_gini(data_set):  # 选择最好的划分方式(按照基尼不纯度）
    num_features = len(data_set[0]) - 1  # 特征数等于每一行的元素数-1（因为最后一个是标签）
    base_gini_impurity = calc_gini_impurity(data_set)  # 求基尼不纯度的原始值
    best_info_gain = 0.0  # 初始化最好的信息增益为0.0
    best_feature = -1  # 初始化最好的分割axis为-1
    for i in range(num_features):  # 遍历各特征
        feat_list = [example[i] for example in data_set]  # 创建一个这个特征的所有值的列表
        unique_vals = set(feat_list)  # 转换为集合
        new_gini_impurity = 0.0  # 新的基尼不纯度初始化为0.0
        for value in unique_vals:  # 遍历特征的值的集合
            sub_data_set = split_data_set(data_set, i, value)  # 尝试划分
            prob = len(sub_data_set) / float(len(data_set))  # 求该划分的占比
            # print('第', i, '维内容为', value, '的该划分的占比为：', prob, '基尼不纯度为：', calc_gini_impurity(sub_data_set))
            new_gini_impurity += prob * calc_gini_impurity(sub_data_set)  # 累加这个划分的占比*这个划分的基尼不纯度
        # print('以第', i, '维划分的基尼不纯度为：', new_gini_impurity)
        info_gain = base_gini_impurity - new_gini_impurity  # 计算信息增益（也就是香农熵的差值）
        # print('以第', i, '维划分的信息增益为：', info_gain)
        if info_gain > best_info_gain:  # 把正在计算的这个和目前最好的相比较
            best_info_gain = info_gain  # 更好的就替换
            best_feature = i  # 最好地划分特征也替换为当前特征
    return best_feature  # 返回最好地划分特征对应的维数


def majority_cnt(class_list):  # 投票表决法获取class_list中出现最多次的标签（解决特征都用完，仍然有不同标签的问题）
    class_count = {}  # 创建标签计数字典
    for vote in class_list:  # 生成标签计数字典的数值
        if vote not in class_count.keys():  # 若标签计数字典中没记录
            class_count[vote] = 0  # 则初始化为0
        class_count[vote] += 1  # 进行一个计数的加1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)  # 对class_count按第一维排序
    return sorted_class_count[0][0]  # 返回计数最多的标签名称


def create_tree(data_set, labels):  # 创建决策树(按照香农熵）
    sub_labels = labels[:]  # 创建一个新的标签列表（解决对原标签列表修改的问题）
    class_list = [example[-1] for example in data_set]  # 创建一个所有特征的列表
    # 叶节点条件1：全条目都是同一种
    if class_list.count(class_list[0]) == len(class_list):  # 若全条目都是同一个特征
        return class_list[0]  # 返回当前特征（叶节点）
    # 叶节点条件2：没有特征可用于划分
    if len(data_set[0]) == 1:  # 若没有特征可用于划分（只剩下一项标签了）
        return majority_cnt(class_list)  # 返回当前列表里最多见的特征（投票法）（叶节点）
    # 分支节点
    best_feat = choose_best_feature_to_split(data_set)  # 获得最好地划分特征编号
    best_feat_label = sub_labels[best_feat]  # 获得最好地划分特征
    my_tree = {best_feat_label: {}}  # 初始化分支节点字典（字典的值也是一部字典，用于装入下层节点）
    del (sub_labels[best_feat])  # 在标签列表里删掉最好特征（划分完后数据列里也没这项了）
    feat_values = [example[best_feat] for example in data_set]  # 创建一个这个最好特征的所有值的列表
    unique_vals = set(feat_values)  # 转换为集合
    for value in unique_vals:  # 对最好特征的所有值进行遍历
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)  # 本分支节点的字典中的值字典里加入为下一层对应的返回值（嵌套）
    return my_tree


def create_tree_gini(data_set, labels):  # 创建决策树(按照基尼不纯度）
    sub_labels = labels[:]  # 创建一个新的标签列表（解决对原标签列表修改的问题）
    class_list = [example[-1] for example in data_set]  # 创建一个所有特征的列表
    # 叶节点条件1：全条目都是同一种
    if class_list.count(class_list[0]) == len(class_list):  # 若全条目都是同一个特征
        return class_list[0]  # 返回当前特征（叶节点）
    # 叶节点条件2：没有特征可用于划分
    if len(data_set[0]) == 1:  # 若没有特征可用于划分（只剩下一项标签了）
        return majority_cnt(class_list)  # 返回当前列表里最多见的特征（投票法）（叶节点）
    # 分支节点
    best_feat = choose_best_feature_to_split_gini(data_set)  # 获得最好地划分特征编号
    best_feat_label = sub_labels[best_feat]  # 获得最好地划分特征
    my_tree = {best_feat_label: {}}  # 初始化分支节点字典（字典的值也是一部字典，用于装入下层节点）
    del (sub_labels[best_feat])  # 在标签列表里删掉最好特征（划分完后数据列里也没这项了）
    feat_values = [example[best_feat] for example in data_set]  # 创建一个这个最好特征的所有值的列表
    unique_vals = set(feat_values)  # 转换为集合
    for value in unique_vals:  # 对最好特征的所有值进行遍历
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)  # 本分支节点的字典中的值字典里加入为下一层对应的返回值（嵌套）
    return my_tree


# #####以上是生成决策树相关函数（来自3.1：决策树的构造.py）#####
# #####以下是绘图函数#####
def plot_node(node_txt, center_pt, parent_pt, node_type):  # 画节点
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt, textcoords='axes fraction', va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


def get_num_leafs(my_tree):  # 获取叶节点的数目
    num_leafs = 0  # 初始化叶节点的数目=0
    first_str = list(my_tree)[0]  # first_str指向第一个index
    second_dict = my_tree[first_str]  # second_dict指向第一个值
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':  # 测试节点的数据类型是否为字典，否则就是叶节点
            num_leafs += get_num_leafs(second_dict[key])  # 累加他子节点的get_num_leafs（嵌套）
        else:
            num_leafs += 1  # 叶节点的话加一
    return num_leafs


# def get_tree_depth(my_tree):  # 获取树的树高（包括叶节点）
#     max_depth = 0  # 初始化树的层数=0
#     first_str = list(my_tree)[0]  # first_str指向第一个index
#     try:
#         second_dict = my_tree[first_str]  # second_dict指向索引为first_str的值
#         for key in second_dict.keys():
#             this_depth = get_tree_depth(second_dict[key]) + 1
#             if this_depth > max_depth:
#                 max_depth = this_depth
#     except TypeError:
#         max_depth = 1
#     return max_depth


def get_tree_depth(my_tree):  # 获取树的树高（实际上不包括叶节点，绘图需要的也是这个）
    max_depth = 0  # 初始化树的层数=0
    first_str = list(my_tree)[0]  # first_str指向第一个index
    second_dict = my_tree[first_str]  # second_dict指向第一个值
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':  # 测试节点的数据类型是否为字典，否则就是叶节点
            this_depth = 1 + get_tree_depth(second_dict[key])  # 深度等于他子节点的get_tree_depth+1（嵌套）
        else:
            this_depth = 1  # 叶节点深度等于1
        if this_depth > max_depth:
            max_depth = this_depth  # 大的话替换现有值
    return max_depth


def retrieve_tree(i):  # 预先存储树的信息（用于测试）
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}, {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return list_of_trees[i]


def plot_mid_text(cntr_pt, parent_pt, txt_string):  # 在父与子节点中间填充文本信息
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid + 0.03, txt_string, va="center", ha="center", rotation=30)


def create_plot(in_tree):  # 画图函数
    matplotlib.rcParams['font.family'] = matplotlib.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文支持，中文字体为简体黑体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plot_tree.totalW = float(get_num_leafs(in_tree))  # 存储树的深度
    plot_tree.totalD = float(get_tree_depth(in_tree))  # 存储树的高度
    plot_tree.xOff = -0.5 / plot_tree.totalW  # 跟踪当前x坐标
    plot_tree.yOff = 1.0  # 跟踪当前y坐标
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()


def plot_tree(my_tree, parent_pt, node_txt):  # 画子树函数
    num_leafs = float(get_num_leafs(my_tree))  # 计算当前树的最大宽度（叶节点个数）
    # depth = get_tree_depth(my_tree)  # 计算当前树的最大高度（树高）
    first_str = list(my_tree)[0]  # first_str指向第一个index
    cntr_pt = (plot_tree.xOff + (1.0 + num_leafs) / 2.0 / plot_tree.totalW, plot_tree.yOff)  # 设置下一节点坐标
    plot_mid_text(cntr_pt, parent_pt, node_txt)  # 在父与子节点中间填充文本信息
    plot_node(first_str, cntr_pt, parent_pt, decision_node)  # 画分支节点
    second_dict = my_tree[first_str]  # second_dict指向第一个值
    plot_tree.yOff -= 1.0 / plot_tree.totalD  # 当前y坐标往下移动一格（1/totalD）
    for key in second_dict.keys():  # 遍历子节点字典里的每个节点
        if type(second_dict[key]).__name__ == 'dict':  # 如果是分支节点
            plot_tree(second_dict[key], cntr_pt, str(key))  # 画他
        else:  # 如果是叶节点
            plot_tree.xOff += 1.0 / plot_tree.totalW  # x坐标往右移动一格（1/totalW）
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntr_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD  # 当前y坐标往上移动一格（1/totalD）


# #####以上是绘图函数#####
# #####以下是决策树使用和存储的函数#####
def classify(input_tree, feat_labels, test_vec):  # 使用决策树的分类函数
    first_str = list(input_tree)[0]  # first_str指向决策树的第一个index（特征名称）
    second_dict = input_tree[first_str]  # second_dict指向决策树的第一个值（字典）
    feat_index = feat_labels.index(first_str)  # feat_index指向first_str在feat_labels列表的第一个匹配项的索引位置（序号）
    key = test_vec[feat_index]  # key指向test_vec向量的序号为feat_index的项
    value_of_feat = second_dict[key]  # value_of_feat指向second_dict字典中索引为key的项的值
    if isinstance(value_of_feat, dict):  # 若value_of_feat是字典
        class_label = classify(value_of_feat, feat_labels, test_vec)  # 递归进行下一步
    else:  # 若value_of_feat不是字典
        class_label = value_of_feat  # 返回value_of_feat
    return class_label


def store_tree(input_tree, filename):  # 存储决策树
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(input_tree, fw)
    fw.close()


def grab_tree(filename):  # 取出决策树
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


# #####以上是决策树使用和存储的函数#####
def file2matrix(filename):  # 文件转矩阵
    file = open(filename)
    file_lines = file.readlines()
    translate_dictionary = {'young': '年轻人', 'presbyopic': '老年人', 'pre': '学龄前', 'myope': '近视眼', 'hyper': '美观需求', 'yes': '是', 'no lenses': '不建议戴\n隐形眼镜', 'normal': '正常', 'no': '否', 'reduced': '偏少', 'soft': '可戴软质\n隐形眼镜', 'hard': '可戴硬质\n隐形眼镜'}  # 字典型
    for key, value in translate_dictionary.items():
        for numOfLines in range(len(file_lines)):
            file_lines[numOfLines] = file_lines[numOfLines].replace(key, value)
    lenses = [inst.strip().split('\t') for inst in file_lines]
    return lenses


# #####运行区域#####
lensesLabels = ['年龄', '需求', '散光水平', '眼泪量']
lensesTree = create_tree(file2matrix(sourceFilePath), lensesLabels)
print('----------进行存取测试测试----------')
store_tree(lensesTree, filePath + '/%s' % fileName)
print('存入决策树为：', lensesTree)
lensesTree = grab_tree(filePath + '/%s' % fileName)
print('取出决策树为：', lensesTree)
# print(lensesTree)
create_plot(lensesTree)
