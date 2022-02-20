import os  # 用于修改当前文件夹
import sys  # 用于获取命令行参数

# #####设置区域#####
# 功能开关
codeSetFilePath = True  # 因为答题要求是在命令行手动进入工作路径，所以这里提供了自动进入工作路径的接口，要用时改为True
cmdParameterFirst = True  # 把这个改为False就不会接受命令行参数了，True的话有写入命令行参数则底下设置的目标文件名会被代替
# 路径设置
filePath = 'D:/Desktop/文件处理作业'  # 此处为作业文件夹的路径（要由此设置的话要把上行改为True）
targetFileName = 'output.txt'  # 此处为程序输出结果的目标文件名（如果cmdParameterFirst==True的话有可能被命令行参数顶替）
rightAnswerName = 'ans.txt'  # 此处为正确的输出文件的文件名（用于自动验证程序输出是否正确）


# 更改相关以上文件名注意保留左右的单引号，确保等号右边为字符串
# 路径的斜杠是反的要注意一下（和平常写路径的方法是不一样的）

# 按行读入列表用函数
def read_by_line(file_node, target_list):
    while True:
        data1 = file_node.readline()
        if data1 == '':
            break
        target_list.append(data1.strip())
        # 测试用语句：print(target_list)
    file_node.close()


# #####执行区域#####
# 获取命令行参数和修改
if cmdParameterFirst and len(sys.argv) > 1:  # 如果有传入的命令行参数
    targetFileName = sys.argv[1]  # 修改输出文件名字
# 调整当前文件夹
print('【当前路径】' + os.getcwd())
if codeSetFilePath:
    print('【使用代码内设置的路径】' + filePath)
    os.chdir(filePath)
    print('【当前路径已经改为】' + os.getcwd())
else:
    print('【不使用代码内设置的路径】')
    print('【当前路径仍为】' + os.getcwd())

# 判断结果是否正确
print('-----开始验证结果是否正确-----')
# 打开文件
ansTXT = open(rightAnswerName, 'r', encoding='utf-8')
print('【打开正确答案文件】' + rightAnswerName)
outputTXT = open(targetFileName, 'r', encoding='utf-8')
print('【打开输出结果文件】' + targetFileName)
# 读入
rightAnswer = []
outputAnswer = []
read_by_line(ansTXT, rightAnswer)
print('【正确答案读取完成】')
read_by_line(outputTXT, outputAnswer)
print('【输出答案读取完成】')
# 关闭文件
ansTXT.close()
print('【关闭正确答案文件】' + rightAnswerName)
outputTXT.close()
print('【关闭输出结果文件】' + targetFileName)
# 进行验证（逐个比对）
for i in range(max(len(rightAnswer), len(outputAnswer))):
    if rightAnswer[i] != outputAnswer[i]:
        print('【输出答案有误】输出本句说明题目代码没写对')
        print('【发现正确条目为】' + rightAnswer[i])
        print('【发现错误条目为】' + outputAnswer[i])
        break
else:
    print('【输出答案正确】输出本句说明题目代码写对了')
