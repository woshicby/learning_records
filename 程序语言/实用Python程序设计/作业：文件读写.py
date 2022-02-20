import os  # 用于修改当前文件夹
import sys  # 用于获取命令行参数

# #####设置区域#####
# 功能开关
codeSetFilePath = True  # 因为答题要求是在命令行手动进入工作路径，所以这里提供了自动进入工作路径的接口，要用时改为True
cmdParameterFirst = True  # 把这个改为False就不会接受命令行参数了，True的话有写入命令行参数则底下设置的目标文件名会被代替
answerTest = True  # 是否进行答案验证？True的话需要指定底下的rightAnswerName
# 路径设置
filePath = 'D:/Desktop/文件处理作业'  # 此处为作业文件夹的路径（要由此设置的话要把上行改为True）
studentsInformationFileName = 'id.txt'  # 此处为题目提供的学生信息文件名
studentsAnswerFileName = 'finalscore.txt'  # 此处为题目提供的学生答题记录文件名
targetFileName = 'output.txt'  # 此处为程序输出结果的目标文件名（如果cmdParameterFirst==True的话有可能被命令行参数顶替）
rightAnswerName = 'ans.txt'  # 此处为正确的输出文件的文件名（用于自动验证程序输出是否正确）


# 更改相关以上文件名注意保留左右的单引号，确保等号右边为字符串


# #####定义函数区域#####
# 算分算题数函数
def calculate_score(answer_record):
    answer_record_divide = answer_record.split('\t')
    # 读取答题数
    try:
        num_of_questions = int(answer_record_divide[2])
    except ValueError:  # 由于pycharm不建议使用裸的except语句，此处写出强制转换int会出现的错误类型为ValueError（值错误）
        num_of_questions = int(answer_record_divide[3])
    # 根据答题数确定分数
    if num_of_questions == 0:
        final_score = 0
    elif num_of_questions == 1:
        final_score = 50
    elif num_of_questions == 2:
        final_score = 60
    else:
        final_score = num_of_questions * 4 + 52
    return str(num_of_questions) + '\t' + str(final_score)


# 按行读入列表用函数
def read_by_line(file_node, target_list):
    while True:
        data1 = file_node.readline()
        if data1 == '':
            break
        target_list.append(data1.strip())
        # 测试用语句：print(target_list)
    file_node.close()


# 输出一行记录用函数
def write_record(target_file, record_id, record_name, num_of_questions_and_score):
    print('【输出一条记录】' + str(record_id) + '\t' + str(record_name) + '\t' + str(num_of_questions_and_score))
    target_file.write(str(record_id) + '\t' + str(record_name) + '\t' + str(num_of_questions_and_score) + '\n')


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

# 打开文件
finalScoreTXT = open(studentsAnswerFileName, 'r', encoding='utf-8')
print('【打开学生答题记录文件】' + studentsAnswerFileName)
idTXT = open(studentsInformationFileName, 'r', encoding='utf-8')
print('【打开学生信息文件】' + studentsInformationFileName)
outputTXT = open(targetFileName, 'w', encoding='utf-8')
print('【打开输出结果文件】' + targetFileName)

# 读入
scores = []
students = []
read_by_line(finalScoreTXT, scores)
print('【学生答题记录读取完成】')
read_by_line(idTXT, students)
print('【学生信息读取完成】')

# 关闭读入文件
finalScoreTXT.close()
print('【关闭学生答题记录文件】' + studentsAnswerFileName)
idTXT.close()
print('【关闭学生信息文件】' + studentsInformationFileName)

# 对学生信息排序
students.sort()
print('【学生信息排序完成】')

# 进行运算和输出
# 结果文件输出第一行
print('【输出标题行】' + '学号\t姓名\t题数\t分数')
outputTXT.write('学号\t姓名\t题数\t分数\n')
# 对每个学生找记录，逐条输出
for student in students:
    student = student.split('\t')
    Id = student[0]
    Name = student[1]
    findFlag = False  # 记录是否找到
    # 测试用语句：print('第'+str(studentId)+"号是【"+str(studentName)+'】')
    for score in scores:
        if Id in score or Name in score:
            # 测试用语句：print(Id, Name, calculate_score(score))
            write_record(outputTXT, Id, Name, calculate_score(score))
            findFlag = True  # 标记找到了
    if not findFlag:  # 没找到，记零分
        write_record(outputTXT, Id, Name, '0\t0')
print('【输出完成】')
outputTXT.close()
print('【关闭输出结果文件】' + targetFileName)

# 判断结果是否正确
if answerTest:
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
            print(rightAnswer[i], outputAnswer[i])
            break
    else:
        print('【输出答案正确】输出本句说明题目代码写对了')
else:
    print('【未开启答案验证功能】')
