import numpy


# #####函数声明区域#####
def load_data_set():  # 生成简单的训练用文章数据集
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 标为1的是侮辱性词语，标为0的不是
    return posting_list, class_vec  # 返回一组文本和他们对应的侮辱性标记


def create_vocabulary_list(data_set):  # 创建单词表
    vocab_set = set([])  # 建立空元组
    for document in data_set:  # 对数据组的每篇文章遍历
        vocab_set = vocab_set | set(document)  # vocab_set和set(document)的并集
    return list(vocab_set)  # 转为列表返回


def set_of_words_to_vec(vocab_list, input_set):  # 根据词表返回数据中的单词出现向量（贝叶斯词集）
    word_appear_vec = [0] * len(vocab_list)  # 初始化返回向量为全0
    for word in input_set:  # 对输入数据的每个单词遍历
        if word in vocab_list:  # 若单词在单词表里
            word_appear_vec[vocab_list.index(word)] = 1  # 标记返回向量的对应元素为1
        else:
            print("这个单词:%s不在单词表里!" % word)
    return word_appear_vec


def bag_of_words_to_vec(vocab_list, input_set):  # 根据词表返回数据中的单词出现次数向量（贝叶斯词袋）
    word_appear_vec = [0] * len(vocab_list)  # 初始化返回向量为全0
    for word in input_set:  # 对输入数据的每个单词遍历
        if word in vocab_list:  # 若单词在单词表里
            word_appear_vec[vocab_list.index(word)] += 1  # 标记返回向量的对应元素为1
        else:
            print("这个单词:%s不在单词表里!" % word)
    return word_appear_vec


def train_naive_bayes_ver0(train_matrix, train_category):  # 第0版本的朴素贝叶斯概率计算器
    num_train_vector = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_vector)  # 得p(侮辱性文章）
    num_vector_of_1 = numpy.zeros(num_words)  # 初始化词汇计数向量为0（对于侮辱性文章）
    num_vector_of_0 = numpy.zeros(num_words)  # 初始化词汇计数向量为0（对于非侮辱性文章）
    total_num_of_1 = 0.0  # 初始化总词汇量为0（对于侮辱性文章）
    total_num_of_0 = 0.0  # 初始化总词汇量为0（对于非侮辱性文章）
    for vecIndex in range(num_train_vector):
        if train_category[vecIndex] == 1:  # 若为侮辱性文章
            num_vector_of_1 += train_matrix[vecIndex]
            total_num_of_1 += sum(train_matrix[vecIndex])
        else:  # 若为非侮辱性文章
            num_vector_of_0 += train_matrix[vecIndex]
            total_num_of_0 += sum(train_matrix[vecIndex])
    p1_vector = num_vector_of_1 / total_num_of_1  # 词汇出现概率向量（对于侮辱性文章）p(出现该词汇|侮辱性文章)
    p0_vector = num_vector_of_0 / total_num_of_0  # 词汇出现概率向量（对于非侮辱性文章）p(出现该词汇|非侮辱性文章)
    return p0_vector, p1_vector, p_abusive


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


def classify_naive_bayes(vec2Classify, p0Vec, p1Vec, pClass1):  # 贝叶斯分类器（根据朴素贝叶斯概率计算器算出的结果）
    abusive_dictionary = {1: '侮辱性文章', 0: '非侮辱性文章'}  # 字典型
    # p(是这个类型|出现这些词汇)=(p(出现这些词汇|是这个类型)p(是这个类型))/p(是这个类型)（计算时分母省略，分子取log）
    p1 = sum(vec2Classify * p1Vec) + numpy.log(pClass1)  # 计算log(p(出现这些词汇|侮辱性文章)p(侮辱性文章))
    p0 = sum(vec2Classify * p0Vec) + numpy.log(1.0 - pClass1)  # 计算log(p(出现这些词汇|非侮辱性文章)p(非侮辱性文章))
    if p1 > p0:
        return abusive_dictionary[1]
    else:
        return abusive_dictionary[0]


# def testing_naive_bayes():  # 贝叶斯分类器测试代码
#     listOfDocuments, listOfClasses = load_data_set()
#     myVocabularyList = create_vocabulary_list(listOfDocuments)
#     trainMatrix = []
#     for Document in listOfDocuments:
#         trainMatrix.append(set_of_words_to_vec(myVocabularyList, Document))
#     p0Vector, P1Vector, POfAbusive = train_naive_bayes_ver1(trainMatrix, listOfClasses)
#     testEntry = ['love', 'my', 'dalmation']
#     thisDoc = numpy.array(set_of_words_to_vec(myVocabularyList, testEntry))
#     print(testEntry, '这篇文章的分类是: ', classify_naive_bayes(thisDoc, p0Vector, P1Vector, POfAbusive))
#     testEntry = ['stupid', 'garbage']
#     thisDoc = numpy.array(set_of_words_to_vec(myVocabularyList, testEntry))
#     print(testEntry, '这篇文章的分类是: ', classify_naive_bayes(thisDoc, p0Vector, P1Vector, POfAbusive))


# #####运行区域#####
listOfDocuments, listOfClasses = load_data_set()
print('----------建立单词列表----------')
myVocabularyList = create_vocabulary_list(listOfDocuments)
print('整个数据组出现的所有单词列表：', myVocabularyList)
trainMatrix = []
print('----------产生朴素贝叶斯词集模型出现向量----------')
for i in range(len(listOfDocuments)):
    trainVec = set_of_words_to_vec(myVocabularyList, listOfDocuments[i])
    print('第%s个文档中的单词出现向量为：' % str(i + 1), trainVec)
    trainMatrix.append(trainVec)
print('----------产生朴素贝叶斯词袋模型出现次数向量----------')
for i in range(len(listOfDocuments)):
    trainVec = bag_of_words_to_vec(myVocabularyList, listOfDocuments[i])
    print('第%s个文档中的单词出现次数向量为：' % str(i + 1), trainVec)
print('----------以下朴素贝叶斯词集模型执行----------')
print('全文章的单词出现矩阵为：\n', trainMatrix)
print('----------第0版本的朴素贝叶斯概率计算器计算结果----------')
p0Vector, P1Vector, POfAbusive = train_naive_bayes_ver0(trainMatrix, listOfClasses)
print('p(出现该词汇|侮辱性文章)为：\n', p0Vector)
print('p(出现该词汇|非侮辱性文章)为：\n', P1Vector)
print('p(侮辱性文章）为：', POfAbusive)
print('----------第1版本的朴素贝叶斯概率计算器计算结果----------')
p0Vector, P1Vector, POfAbusive = train_naive_bayes_ver1(trainMatrix, listOfClasses)
print('log(p(出现该词汇|侮辱性文章))为：\n', p0Vector)
print('log(p(出现该词汇|非侮辱性文章))为：\n', P1Vector)
print('p(侮辱性文章）为：', POfAbusive)
# testing_naive_bayes()
print('----------进行分类测试----------')
testEntry = ['love', 'my', 'dalmation']
thisDoc = numpy.array(set_of_words_to_vec(myVocabularyList, testEntry))
print(testEntry, '这篇文章的分类是: ', classify_naive_bayes(thisDoc, p0Vector, P1Vector, POfAbusive))
testEntry = ['stupid', 'garbage']
thisDoc = numpy.array(set_of_words_to_vec(myVocabularyList, testEntry))
print(testEntry, '这篇文章的分类是: ', classify_naive_bayes(thisDoc, p0Vector, P1Vector, POfAbusive))
