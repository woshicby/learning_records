# 摄氏华氏互转
temStr = input("请输入带有符号的温度值")
if 'F' in temStr or 'f' in temStr:
    print("转换后的温度是" + str((float(temStr[0:-1]) - 32) / 1.8) + 'C')
elif temStr[-1] in 'cC':
    print("转换后的温度是" + str(eval(temStr[0:-1]) * 1.8 + 32) + 'F')
else:
    print("格式输入错误")
