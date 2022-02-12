import turtle  # 画图用turtle包


def snow(n, size):  # n为阶数，size为长度，从当前起点出发，在当前方向画一个长度为size，阶为n的雪花曲线
	if n == 0:
		turtle.fd(size)  # 笔沿当前方向前进size
	else:
		for angle in [0, 60, -120, 60]:  # 对列表中的每个元素angle
			turtle.left(angle)  # 笔左转angle度，turtle.lt(angle)也行
			snow(n - 1, size / 3)


turtle.setup(1000, 600)
turtle.speed(0)
turtle.delay(0)
# 窗口缺省位于屏幕正中间，宽1000*600像素，窗口中央坐标（0，0）
# 初始笔的前进方向为0度。正东方为0度，正北为90度
turtle.penup()  # 抬起笔
turtle.goto(-400, -100)  # 把笔移动到-400,-100位置
turtle.pendown()  # 放下笔
turtle.pensize(3)  # 笔的粗细为3
snow(10, 800)  # 绘制长度为800，阶为5的雪花曲线，方向水平
turtle.done()  # 保持绘图窗口
