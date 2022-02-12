import turtle  # 画图用turtle包


def strange(n, size, side):  # n为阶数，size为长度，从当前起点出发，在当前方向画一个长度为size，阶为n的雪花曲线
    each_angle = -360 / side
    if n == 0:
        for i in range(side):  # 对列表中的每个元素angle
            turtle.fd(size)  # 笔沿当前方向前进size
            turtle.left(each_angle)  # 笔左转angle度，turtle.lt(angle)也行
    else:
        for i in range(side):  # 对列表中的每个元素angle
            strange(n - 1, size / 2, side)
            turtle.fd(size)
            turtle.left(each_angle)  # 笔左转angle度，turtle.lt(angle)也行


turtle.setup(1000, 1000)
turtle.speed(0)
# 窗口缺省位于屏幕正中间，宽1000*1000像素，窗口中央坐标（0，0）
# 初始笔的前进方向为0度。正东方为0度，正北为90度
turtle.penup()  # 抬起笔
turtle.goto(-300, -300)  # 把笔移动到-400,-331位置
turtle.pendown()  # 放下笔
turtle.pensize(1)  # 笔的粗细为1
level = 2
sides = 50
turtle.left(180 - 360 / sides)  # 调整起始方向60度
strange(level, 40, sides)  # 绘制长度为800，阶为level的奇异三角形，方向水平
turtle.done()  # 保持绘图窗口
