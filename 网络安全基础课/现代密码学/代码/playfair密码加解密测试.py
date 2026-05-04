# 添加x
def add_x(string):
    i = 0
    string_ready = ""
    while i < len(string) - 1:
        if string[i] == string[i + 1]:  # 连续两个一样，中间加一个x
            string_ready += string[i] + "x"
            i += 1
        else:  # 两个字母不一样，正常加入
            string_ready += string[i] + string[i + 1]
            i += 2
    if i == len(string) - 1:  # 最后一个落单，补x
        string_ready += string[i] + "x"
    return string_ready


# 去除x（无法完全）
def sub_x(string):
    string_list = list(string)
    for i in range(len(string) - 2, 1, -1):  # 遍历检查
        if string_list[i] == "x" and string_list[i - 1] == string_list[i + 1]:  # x前后两个字母一样，去除x
            string_list[i] = ""
    if string_list[-1] == "x":  # 最后一个是x，去除x
        del string_list[-1]
    # print("去x前", string)
    # print("去x后", string_list)
    return "".join(string_list)


# 建立密码表
def build(key):
    key_metrix = ""
    for i in key + "abcdefghiklmnopqrstuvwxyz":
        if i not in key_metrix:
            key_metrix += i
    # print("密钥矩阵", key_metrix)
    return key_metrix


def encryption(plain_text, key):
    plain_text = plain_text.replace("j", "i").replace(" ", "")  # 合并i、j，去除空格
    key = key.replace("j", "i").replace(" ", "")  # 合并i、j，去除空格
    cipher_text = ""  # 初始化密文
    table = build(key)
    plain_text_ready = add_x(plain_text)
    # print("准备好的明文", plain_text_ready)
    for i in range(0, len(plain_text_ready) - 1, 2):
        x1 = table.index(plain_text_ready[i]) // 5
        y1 = table.index(plain_text_ready[i]) % 5
        x2 = table.index(plain_text_ready[i + 1]) // 5
        y2 = table.index(plain_text_ready[i + 1]) % 5
        if y1 == y2:  # 同列
            x1 = (x1 + 1) % 5
            x2 = (x2 + 1) % 5
        elif x1 == x2:  # 同行
            y1 = (y1 + 1) % 5
            y2 = (y2 + 1) % 5
        else:
            x1, x2 = x2, x1  # 横向取对角
        cipher_text += table[x1 * 5 + y1] + table[x2 * 5 + y2]  # 加入密文
    return cipher_text


# 解密似乎有多种答案， 因为加密时添加 x ， 在解密时不知道是本生就有，还是加密过程中添加的
def decryption(cipher_text, key):
    cipher_text = cipher_text.replace("j", "i").replace(" ", "")
    key = key.replace("j", "i").replace(" ", "")
    plain_text = ""
    table = build(key)
    for i in range(0, len(cipher_text) - 1, 2):
        x1 = table.index(cipher_text[i]) // 5
        y1 = table.index(cipher_text[i]) % 5
        x2 = table.index(cipher_text[i + 1]) // 5
        y2 = table.index(cipher_text[i + 1]) % 5
        if y1 == y2:  # 同列
            x1 = (x1 - 1) % 5
            x2 = (x2 - 1) % 5
        elif x1 == x2:  # 同行
            y1 = (y1 - 1) % 5
            y2 = (y2 - 1) % 5
        else:
            x1, x2 = x2, x1  # 横向取对角
        plain_text += table[x1 * 5 + y1] + table[x2 * 5 + y2]  # 加入密文
    return sub_x(plain_text)


word = input("输入要处理的playfair密码的密/明文：")
keys = input("输入playfair密码密钥：")
print("-----用" + keys + "对" + word + "加密后是：-----\n" + encryption(word, keys))
print("-----用" + keys + "对" + word + "解密后是：-----\n" + decryption(word, keys))
