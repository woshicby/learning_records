import math  # 导入数学库
# 020:求一元二次方程的根
a = input()
c = float(a.split()[2])
b = float(a.split()[1])
a = float(a.split()[0])
delta = b ** 2 - 4 * a * c
Imaginary = math.sqrt(abs(delta)) / 2 / a  # 求虚部
if -b / 2 / a == 0:  # 求实部（消除0前面的负号）
    Real = 0
else:
    Real = -b / 2 / a
if delta == 0:
    print("x1=x2=" + "%.5f" % Real)
elif delta > 0:
    print("x1=" + "%.5f" % (Real + Imaginary) + ";x2=" + "%.5f" % (Real - Imaginary))
else:
    print("x1=" + "%.5f" % Real + "+" + "%.5f" % Imaginary + "i;x2=" + "%.5f" % Real + "-" + "%.5f" % Imaginary + "i")
