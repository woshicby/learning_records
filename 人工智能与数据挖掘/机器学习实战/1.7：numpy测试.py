import numpy

randArray = numpy.random.rand(4, 4)
print('随机数组\n', randArray)
randMat = numpy.mat(randArray)
invRandMat = randMat.I
myEye = invRandMat * randMat
print('随机矩阵\n', randMat)
print('随机矩阵的转置\n', randMat.T)
print('随机矩阵的逆\n', invRandMat)
print('矩阵与逆的乘积\n', myEye)
print('单位矩阵\n', numpy.eye(4))
print('求逆的误差\n', myEye - numpy.eye(4))
