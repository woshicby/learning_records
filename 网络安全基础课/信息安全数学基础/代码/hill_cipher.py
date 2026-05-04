import numpy
from euclid_inverse import inverse  # 导入求逆


def alphabet_to_unicode(alphabet):  # 将无论大小写字母转化为它在字母表中的排位（非英文字母直接返回unicode值）
    if 'a' <= alphabet <= 'z':  # 小写字母
        return ord(alphabet) - ord('a') + 1
    elif 'A' <= alphabet <= 'Z':  # 大写字母
        return ord(alphabet) - ord('A') + 1
    else:  # 其他字符返回unicode值
        return ord(alphabet)


# 将文本转换为ascii码值矩阵，行数与加密密钥保持一致
def char2ascii2(text, row, m):
    # 输入：明文plaintext，加密矩阵的行数row,模m
    l1 = [0] * row
    l2 = []
    for i in range(len(text)):
        j = i % row
        if i > 0 and i % row == 0:
            l2.append(l1)
            l1 = [0, 0, 0]
        l1[j] = ord(text[i])
    l2.append(l1)
    m1 = numpy.array(l2)
    m1 = numpy.reshape(m1, (m1.shape[1], m1.shape[0]))
    m1 = m1 % 26
    return m1


# 将矩阵(二维数组)ascii码转字符
def ascii2_char(array1):
    text = ''
    row = array1.shape[0]
    col = array1.shape[1]
    for i in range(row):
        for j in range(col):
            text = text + chr(array1[i][j])
    return text


def input_encrypt_numpy_array():  # 输入加密矩阵，判断是否可逆，不可逆让他一直输，输到可以为止
    row_num = int(input("输入希尔密码密钥矩阵行数：（需为整数）"))
    key_list = []
    for i in range(row_num):
        row_list = input("请输入加密密钥矩阵第" + str(i + 1) + "行（两个值中间敲空格）").split(' ')
        if len(row_list) == row_num:  # 输入数量对的话
            for n in range(len(row_list)):  # 逐个转int
                row_list[n] = int(row_list[n])
            key_list.append(row_list)
        else:
            print("该行数量不对，请重修输入")
            del key_list, row_num, row_list  # 这一层的相关变量已经不能要了
            return input_encrypt_numpy_array()
    encrypt_numpy_array = numpy.array(key_list)  # 转numpy.array
    det = int(round(numpy.linalg.det(encrypt_numpy_array)))  # 求一下行列式（四舍五入修正numpy误差）
    if det == 0 or inverse(26, det) == "不存在":  # 行列式=0/不可逆
        del key_list, row_num, row_list, det  # 这一层的相关变量已经不能要了
        return input_encrypt_numpy_array()
    else:  # 如果可逆
        return encrypt_numpy_array, det


def inverse_metrix(matrix, det):  # 求逆矩阵（模26）
    inver = numpy.linalg.inv(matrix)  # 普通的求逆
    star = numpy.around(inver * det % 26)  # 倒回伴随矩阵
    return inver(26, det) * star.astype(int) % 26


def encryption(plain_text, encrypt_key):
    # 转数字串
    plain_text_num = []
    for i in range(len(plain_text)):
        plain_text_num.append(alphabet_to_unicode(i) - 1)
    while len(plain_text_num) % encrypt_key.shape[0] != 0:
        plain_text_num.append(25)
    numpy.array(plain_text_num)


word = input("输入要处理的改进的希尔密码的密/明文：")
encrypt_numpy, det_encrypt_numpy = input_encrypt_numpy_array()

print(encrypt_numpy)
print(det_encrypt_numpy)
print(inverse_metrix(encrypt_numpy, det_encrypt_numpy))
# print("-----用" + keys + "对" + word + "加密后是：-----\n" + encryption(word, encrypt_numpy))
# print("-----用" + keys + "对" + word + "解密后是：-----\n" + decryption(word, inverse_metrix(encrypt_numpy, det_encrypt_numpy)))
