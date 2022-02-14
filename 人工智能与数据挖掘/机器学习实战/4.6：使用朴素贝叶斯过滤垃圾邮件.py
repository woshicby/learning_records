import numpy

# #####设置区域#####
spamEmail = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch04\email\spam'  # 训练数据源文件路径
hamEmail = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch04\email\ham'  # 测试数据源文件路径
abusiveDictionary = {1: '侮辱性文章', 0: '非侮辱性文章'}  # 字典型


# #####函数声明区域#####


def create_vocabulary_list(data_set):  # 创建单词表
    vocab_set = set([])  # 建立空元组
    for document in data_set:  # 对数据组的每篇文章遍历
        vocab_set = vocab_set | set(document)  # vocab_set和set(document)的并集
    return list(vocab_set)  # 转为列表返回


def train_naive_bayes_ver1(train_matrix, train_category):  # 第1版本的朴素贝叶斯概率计算器
    num_train_vector = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_vector)  # 得p(侮辱性文章）
    # 初始化与train_naive_bayes_ver0不同（因为之后要进行概率的相乘）
    num_vector_of_1 = numpy.ones(num_words)  # 初始化词汇计数向量为1（对于侮辱性文章）
    num_vector_of_0 = numpy.ones(num_words)  # 初始化词汇计数向量为1（对于非侮辱性文章）
    total_num_of_1 = 1.0  # 初始化总词汇量为1（对于侮辱性文章）
    total_num_of_0 = 1.0  # 初始化总词汇量为1（对于非侮辱性文章）
    for vecIndex in range(num_train_vector):
        if train_category[vecIndex] == 1:  # 若为侮辱性文章
            num_vector_of_1 += train_matrix[vecIndex]
            total_num_of_1 += sum(train_matrix[vecIndex])
        else:  # 若为非侮辱性文章
            num_vector_of_0 += train_matrix[vecIndex]
            total_num_of_0 += sum(train_matrix[vecIndex])
    # 求log把乘法转换为加法，防止多个小数相乘的下溢出
    p1_vector_logged = numpy.log(num_vector_of_1 / total_num_of_1)  # 词汇出现概率向量（对于侮辱性文章）p(出现该词汇|侮辱性文章)
    p0_vector_logged = numpy.log(num_vector_of_0 / total_num_of_0)  # 词汇出现概率向量（对于非侮辱性文章）p(出现该词汇|非侮辱性文章)
    return p0_vector_logged, p1_vector_logged, p_abusive


def bag_of_words_to_vec(vocab_list, input_set):  # 根据词表返回数据中的单词出现次数向量（贝叶斯词袋）
    word_appear_vec = [0] * len(vocab_list)  # 初始化返回向量为全0
    for word in input_set:  # 对输入数据的每个单词遍历
        if word in vocab_list:  # 若单词在单词表里
            word_appear_vec[vocab_list.index(word)] += 1  # 标记返回向量的对应元素为1
        else:
            print("这个单词:%s不在单词表里!" % word)
    return word_appear_vec


def classify_naive_bayes(vec2classify, p0vec, p1vec, p_class1):  # 贝叶斯分类器（根据朴素贝叶斯概率计算器算出的结果）
    # p(是这个类型|出现这些词汇)=(p(出现这些词汇|是这个类型)p(是这个类型))/p(是这个类型)（计算时分母省略，分子取log）
    p1 = sum(vec2classify * p1vec) + numpy.log(p_class1)  # 计算log(p(出现这些词汇|侮辱性文章)p(侮辱性文章))
    p0 = sum(vec2classify * p0vec) + numpy.log(1.0 - p_class1)  # 计算log(p(出现这些词汇|非侮辱性文章)p(非侮辱性文章))
    if p1 > p0:
        return 1
    else:
        return 0


def text_parse(big_string):  # 输入长字符串，输出单词列表
    import re
    list_of_tokens = re.split(r'\W+', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_test():  # 垃圾邮件过滤测试代码
    doc_list = []
    class_list = []
    full_text = []
    # 读入数据
    for i in range(1, 26):
        word_list = text_parse(open(spamEmail + '/%d.txt' % i, encoding="ISO-8859-1").read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(open(hamEmail + '/%d.txt' % i, encoding="ISO-8859-1").read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocabulary_list(doc_list)  # 创建单词表
    training_set = range(50)
    test_set = []  # 创建测试组
    for i in range(10):  # 抽十个作为测试组
        rand_index = int(numpy.random.uniform(0, len(training_set)))  # 抽取
        test_set.append(training_set[rand_index])  # 加入测试组
        del (list(training_set)[rand_index])  # 从训练组删除
    train_mat = []
    train_classes = []
    for docIndex in training_set:  # 组建训练数据
        train_mat.append(bag_of_words_to_vec(vocab_list, doc_list[docIndex]))
        train_classes.append(class_list[docIndex])
    p0_v, p1_v, p_spam = train_naive_bayes_ver1(numpy.array(train_mat), numpy.array(train_classes))
    error_count = 0
    for docIndex in test_set:  # 进行测试
        word_vector = bag_of_words_to_vec(vocab_list, doc_list[docIndex])
        if classify_naive_bayes(numpy.array(word_vector), p0_v, p1_v, p_spam) != class_list[docIndex]:
            error_count += 1
            print("【分类错误】", doc_list[docIndex])
    print('错误率为: ', float(error_count) / len(test_set))


spam_test()
