# Created on Oct 19, 2010/创建于2010年10月19日
# Translated on Feb 14, 2022/翻译于2022年2月14日
#
# @author/作者: Peter
# @translator/翻译: woshicby
# Ps.This function is modified to fit PEP 8 standard, and I have added Chinese annotations.
#    程序已经修改到符合PEP 8标准，并添加了中文注释
#    Some of them are not translated because I didn't use them
#    有的没翻，因为没用到
import numpy

# #####设置区域#####
abusiveDictionary = {1: '侮辱性文章', 0: '非侮辱性文章'}  # 字典型
spamEmail = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch04\email\spam'  # 普通邮件源文件路径
hamEmail = r'D:\Desktop\新建文件夹\machinelearninginaction3x-master\Ch04\email\ham'  # 垃圾邮件源文件路径


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


def classify_naive_bayes(vec2classify, p0vec, p1vec, p_class1):  # 贝叶斯分类器（根据朴素贝叶斯概率计算器算出的结果）
    # p(是这个类型|出现这些词汇)=(p(出现这些词汇|是这个类型)p(是这个类型))/p(是这个类型)（计算时分母省略，分子取log）
    p1 = sum(vec2classify * p1vec) + numpy.log(p_class1)  # 计算log(p(出现这些词汇|侮辱性文章)p(侮辱性文章))
    p0 = sum(vec2classify * p0vec) + numpy.log(1.0 - p_class1)  # 计算log(p(出现这些词汇|非侮辱性文章)p(非侮辱性文章))
    if p1 > p0:
        return 1
    else:
        return 0


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


'''
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    import feedparser
    docList = [];
    classList = [];
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen);
    testSet = []  # create test set
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (list(trainingSet)[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = [];
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
'''
