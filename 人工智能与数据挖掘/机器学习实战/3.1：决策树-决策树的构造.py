import math
import operator


def create_data_set():  # 生成简单的鱼鉴定数据集
    data_set = [[1, 1, '是'],
                [1, 1, '是'],
                [1, 0, '不是'],
                [0, 1, '不是'],
                [0, 1, '不是']]
    labels = ['不浮出水面', '有脚蹼']
    # 更改为离散值
    return data_set, labels


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
            best_feature = i  # 最好的划分特征也替换为当前特征
    return best_feature  # 返回最好的划分特征对应的维数


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
            best_feature = i  # 最好的划分特征也替换为当前特征
    return best_feature  # 返回最好的划分特征对应的维数


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
    best_feat = choose_best_feature_to_split(data_set)  # 获得最好的划分特征编号
    best_feat_label = sub_labels[best_feat]  # 获得最好的划分特征
    my_tree = {best_feat_label: {}}  # 初始化分支节点字典（字典的值也是一个字典，用于装入下层节点）
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
    best_feat = choose_best_feature_to_split_gini(data_set)  # 获得最好的划分特征编号
    best_feat_label = sub_labels[best_feat]  # 获得最好的划分特征
    my_tree = {best_feat_label: {}}  # 初始化分支节点字典（字典的值也是一个字典，用于装入下层节点）
    del (sub_labels[best_feat])  # 在标签列表里删掉最好特征（划分完后数据列里也没这项了）
    feat_values = [example[best_feat] for example in data_set]  # 创建一个这个最好特征的所有值的列表
    unique_vals = set(feat_values)  # 转换为集合
    for value in unique_vals:  # 对最好特征的所有值进行遍历
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)  # 本分支节点的字典中的值字典里加入为下一层对应的返回值（嵌套）
    return my_tree


myDat, Labels = create_data_set()
print('当前数据集为：\n', myDat, '\n当前标签集为：\n', Labels)
print('当前香农熵为：', calc_shannon_ent(myDat))
print('当前基尼不纯度为：', calc_gini_impurity(myDat))
print('----------修改数据看看----------')
myDat[0][-1] = '可能是'
print('当前数据集为：\n', myDat, '\n当前标签集为：\n', Labels)
print('当前香农熵为：', calc_shannon_ent(myDat))
print('当前基尼不纯度为：', calc_gini_impurity(myDat))
print('----------进行一次划分----------')
print('split_data_set(myDat, 0, 1)的结果为：\n', split_data_set(myDat, 0, 1), '\n'+'split_data_set(myDat, 0, 0)的结果为：\n', split_data_set(myDat, 0, 0))
print('----------选择一次最好划分方式----------')
print('按香农熵选择的最好的划分特征对应的维数为：', choose_best_feature_to_split(myDat))
print('按基尼不纯度选择的最好的划分特征对应的维数为：', choose_best_feature_to_split_gini(myDat))
print('----------生成决策树----------')
print('按香农熵生成的决策树为：', create_tree(myDat, Labels))
print('按基尼不纯度生成的决策树为：', create_tree_gini(myDat, Labels))
