# Created on Jan 8, 2011/创建于2011年1月8日
# Translated on Feb 23, 2022/翻译于2022年2月23日
#
# @author/作者:Peter
# @translator/翻译: woshicby
# Ps.This function is modified to fit PEP 8 standard, and I have added Chinese annotations.
#    程序已经修改到符合PEP 8标准，并添加了中文注释
#    Some modules are not used so keep them intact
#    有的模块没用到，就保持原样
import numpy


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
    return data_mat, label_mat


def standard_regression(x_arr, y_arr):  # 线性回归
    x_mat = numpy.mat(x_arr)
    y_mat = numpy.mat(y_arr).T
    xt_x = x_mat.T * x_mat
    if numpy.linalg.det(xt_x) == 0.0:
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


def locally_weighted_linear_regression(test_point, x_arr, y_arr, k=1.0):  # 局部加权线性回归（高斯核）
    x_mat = numpy.mat(x_arr)
    y_mat = numpy.mat(y_arr).T
    m = numpy.shape(x_mat)[0]
    weights = numpy.mat(numpy.eye(m))  # 创建对角矩阵
    for j in range(m):  # 该循环创建权重矩阵（指数级衰减）
        diff_mat = test_point - x_mat[j, :]  #
        weights[j, j] = numpy.exp(diff_mat * diff_mat.T / (-2.0 * k ** 2))
    xt_x = x_mat.T * (weights * x_mat)
    if numpy.linalg.det(xt_x) == 0.0:  # 如果x转置乘x的行列式等于0
        print("是奇异矩阵, 不能求逆")
        return
    ws = xt_x.I * (x_mat.T * (weights * y_mat))
    return test_point * ws


def locally_weighted_linear_regression_test(test_arr, x_arr, y_arr, k=1.0):  # 遍历所有测试点并计算局部线性回归的结果
    m = numpy.shape(test_arr)[0]
    y_hat = numpy.zeros(m)
    for i in range(m):
        y_hat[i] = locally_weighted_linear_regression(test_arr[i], x_arr, y_arr, k)
    return y_hat


def draw_line_and_point(x_arr, y_arr, y_hat):  # 画数据点和线（8.2）
    x_mat = numpy.mat(x_arr)
    sort_index = x_mat[:, 1].argsort(0)
    x_sort = x_mat[sort_index][:, 0, :]
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = matplotlib.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文支持，中文字体为简体黑体
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_sort[:, 1], y_hat[sort_index])
    ax.scatter(x_mat[:, 1].flatten().A[0], numpy.mat(y_arr).T.flatten().A[0], s=2, c='red')
    plt.show()


def rss_error(y_arr, y_hat_arr):  # 返回一个误差值
    return ((y_arr - y_hat_arr) ** 2).sum()


def ridge_regression(x_mat, y_mat, lam=0.2):  # 岭回归
    xt_x = x_mat.T * x_mat
    denom = xt_x + numpy.eye(numpy.shape(x_mat)[1]) * lam  # 加上‘岭’
    if numpy.linalg.det(denom) == 0.0:
        print("奇异矩阵，无法求逆")
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws


def ridge_test(x_arr, y_arr):  # 岭回归测试器
    x_mat = numpy.mat(x_arr)
    y_mat = numpy.mat(y_arr).T
    y_mean = numpy.mean(y_mat, 0)  # 计算平均值，返回均值的行矩阵
    y_mat = y_mat - y_mean  # 为了消除X0，减去Y的平均值
    # 标准化x
    x_means = numpy.mean(x_mat, 0)  # 计算平均值，之后要减掉它
    x_var = numpy.var(x_mat, 0)  # 计算xi的方差，之后要除以它
    x_mat = (x_mat - x_means) / x_var
    num_test_pts = 30  # 蛮算30步
    w_mat = numpy.zeros((num_test_pts, numpy.shape(x_mat)[1]))  # 初始化一个记录每一步的系数的矩阵
    for i in range(num_test_pts):
        ws = ridge_regression(x_mat, y_mat, numpy.exp(i - 10))
        w_mat[i, :] = ws.T
    return w_mat


def draw_regression_coefficient(ridge_weights):  # 画图相关函数（8.4.1）
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_weights)
    plt.show()


def regularize(x_mat):  # 逐列标准化
    in_mat = x_mat.copy()
    in_means = numpy.mean(in_mat, 0)  # 计算平均值，之后要减掉它
    in_var = numpy.var(in_mat, 0)  # 计算xi的方差，之后要除以它
    in_mat = (in_mat - in_means) / in_var
    return in_mat


def stage_wise(x_arr, y_arr, eps=0.01, num_it=100):
    x_mat = numpy.mat(x_arr)
    y_mat = numpy.mat(y_arr).T
    y_mean = numpy.mean(y_mat, 0)
    y_mat = y_mat - y_mean  # 也能标准化ys，但是会得到更小的coef
    x_mat = regularize(x_mat)
    m, n = numpy.shape(x_mat)
    return_mat = numpy.zeros((num_it, n))  # 初始化一个记录每一代系数的矩阵
    ws = numpy.zeros((n, 1))  # ws记录当代的系数
    # ws_test = ws.copy()
    ws_max = ws.copy()  # 初始化一个存储最大值的列向量
    for i in range(num_it):  # 开始迭代num_it次
        # print(ws.T)
        lowest_error = numpy.inf  # 初始化最小错误值为无穷大
        for j in range(n):  # 遍历每个特征
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps * sign
                y_test = x_mat * ws_test
                rss_e = rss_error(y_mat.A, y_test.A)
                if rss_e < lowest_error:  # 如果错误更小的话就更新
                    lowest_error = rss_e
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T
    return return_mat  # , ws  # 返回的是系数的历史记录矩阵，若要返回最优值，得返回ws
# 后续是8.6用的函数，没用到就没翻译

# def scrapePage(inFile,outFile,yr,numPce,origPrc):
#    from BeautifulSoup import BeautifulSoup
#    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
#    soup = BeautifulSoup(fr.read())
#    i=1
#    currentRow = soup.findAll('table', r="%d" % i)
#    while(len(currentRow)!=0):
#        title = currentRow[0].findAll('a')[1].text
#        lwrTitle = title.lower()
#        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#            newFlag = 1.0
#        else:
#            newFlag = 0.0
#        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#        if len(soldUnicde)==0:
#            print "item #%d did not sell" % i
#        else:
#            soldPrice = currentRow[0].findAll('td')[4]
#            priceStr = soldPrice.text
#            priceStr = priceStr.replace('$','') #strips out $
#            priceStr = priceStr.replace(',','') #strips out ,
#            if len(soldPrice)>1:
#                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
#            print "%s\t%d\t%s" % (priceStr,newFlag,title)
#            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
#        i += 1
#        currentRow = soup.findAll('table', r="%d" % i)
#    fw.close()
#
# from time import sleep
# import json
# import urllib.request
#
#
# def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
#     sleep(10)
#     myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
#     searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
#     pg = urllib.request.urlopen(searchURL)
#     retDict = json.loads(pg.read())
#     for i in range(len(retDict['items'])):
#         try:
#             currItem = retDict['items'][i]
#             if currItem['product']['condition'] == 'new':
#                 newFlag = 1
#             else:
#                 newFlag = 0
#             listOfInv = currItem['product']['inventories']
#             for item in listOfInv:
#                 sellingPrice = item['price']
#                 if sellingPrice > origPrc * 0.5:
#                     print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
#                     retX.append([yr, numPce, newFlag, origPrc])
#                     retY.append(sellingPrice)
#         except:
#             print('problem with item %d' % i)
#
#
# def setDataCollect(retX, retY):
#     searchForSet(retX, retY, 8288, 2006, 800, 49.99)
#     searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
#     searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
#     searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
#     searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
#     searchForSet(retX, retY, 10196, 2009, 3263, 249.99)
#
#
# def crossValidation(xArr, yArr, numVal=10):
#     m = len(yArr)
#     indexList = range(m)
#     errorMat = numpy.zeros((numVal, 30))  # create error mat 30columns numVal rows
#     for i in range(numVal):
#         trainX = []
#         trainY = []
#         testX = []
#         testY = []
#         numpy.random.shuffle(indexList)
#         for j in range(m):  # create training set based on first 90% of values in indexList
#             if j < m * 0.9:
#                 trainX.append(xArr[indexList[j]])
#                 trainY.append(yArr[indexList[j]])
#             else:
#                 testX.append(xArr[indexList[j]])
#                 testY.append(yArr[indexList[j]])
#         wMat = ridge_test(trainX, trainY)  # get 30 weight vectors from ridge
#         for k in range(30):  # loop over all of the ridge estimates
#             matTestX = numpy.mat(testX)
#             matTrainX = numpy.mat(trainX)
#             meanTrain = numpy.mean(matTrainX, 0)
#             varTrain = numpy.var(matTrainX, 0)
#             matTestX = (matTestX - meanTrain) / varTrain  # regularize test with training params
#             yEst = matTestX * numpy.mat(wMat[k, :]).T + numpy.mean(trainY)  # test ridge results and store
#             errorMat[i, k] = rss_error(yEst.T.A, numpy.array(testY))
#             # print errorMat[i,k]
#     meanErrors = numpy.mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
#     minMean = float(min(meanErrors))
#     bestWeights = wMat[numpy.nonzero(meanErrors == minMean)]
#     # can unregularize to get model
#     # when we regularized we wrote Xreg = (x-meanX)/var(x)
#     # we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
#     xMat = numpy.mat(xArr)
#     yMat = numpy.mat(yArr).T
#     meanX = numpy.mean(xMat, 0)
#     varX = numpy.var(xMat, 0)
#     unReg = bestWeights / varX
#     print("the best model from Ridge Regression is:\n", unReg)
#     print("with constant term: ", -1 * sum(numpy.multiply(meanX, unReg)) + numpy.mean(yMat))
