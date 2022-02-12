# Created on Oct 14, 2010/创建于2010年10月14日
# Translated on Feb 11, 2022/翻译于2022年2月11日
# @author/作者: Peter Harrington
# @translator/翻译: woshicby
# Ps.This function is modified to fit PEP 8 standard, and I have added Chinese annotations.
#    程序已经修改到符合PEP 8标准，并添加了中文注释
#    Provides Chinese support for create_plot(in_tree).
#    对树状图的绘制函数提供了中文支持
import matplotlib
import matplotlib.pyplot as plt

# #####设置区域#####
# 定义文本框和箭头格式
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# #####函数声明区域#####
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


def plot_node(node_txt, center_pt, parent_pt, node_type):  # 画节点
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt, textcoords='axes fraction', va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


def plot_mid_text(cntr_pt, parent_pt, txt_string):  # 在父与子节点中间填充文本信息
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)


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


def create_plot(in_tree):  # 画图函数
    matplotlib.rcParams['font.family'] = matplotlib.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文支持，中文字体为简体黑体
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


# def create_plot():  # 测试画节点函数用的create_plot
#     matplotlib.rcParams['font.family'] = matplotlib.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文支持，中文字体为简体黑体
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     create_plot.ax1 = plt.subplot(111, frameon=False)  # ticks for demo puropses
#     plot_node('分支节点', (0.5, 0.1), (0.1, 0.5), decision_node)
#     plot_node('叶节点', (0.8, 0.1), (0.3, 0.8), leaf_node)
#     plt.show()


def retrieve_tree(i):  # 预先存储树的信息（用于测试）
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}, {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return list_of_trees[i]


# #####运行区域#####
# 绘图
create_plot(retrieve_tree(0))
