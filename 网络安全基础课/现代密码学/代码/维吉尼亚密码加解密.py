def next_index(index, index_amount):  # 循环变换的下一个序号（范围：0~总个数-1）
    if index < index_amount - 1:  # index未越界
        return index + 1
    else:
        return 0


def alphabet_to_unicode(alphabet):  # 将无论大小写字母转化为它在字母表中的排位（非英文字母直接返回unicode值）
    if 'a' <= alphabet <= 'z':  # 小写字母
        return ord(alphabet) - ord('a') + 1
    elif 'A' <= alphabet <= 'Z':  # 大写字母
        return ord(alphabet) - ord('A') + 1
    else:  # 其他字符返回unicode值
        return ord(alphabet)


def encryption(plain_text, key):  # 右移加密
    cipher_text = ""  # 初始化密文
    key_index = 0  # 初始化当前密钥序号
    for character in plain_text:
        if 'a' <= character <= 'z':  # 小写字母
            cipher_text += chr(ord('a') + (ord(character) - ord('a') + alphabet_to_unicode(key[key_index])) % 26)
        elif 'A' <= character <= 'Z':  # 大写字母
            cipher_text += chr(ord('A') + (ord(character) - ord('A') + alphabet_to_unicode(key[key_index])) % 26)
        else:  # 其他字符不处理
            cipher_text += character
        key_index = next_index(key_index, len(key))  # 下一个密钥序号
    return cipher_text


def decryption(cipher_text, key):  # 左移解密
    plain_text = ""  # 初始化明文
    key_index = 0  # 初始化当前密钥序号
    for character in cipher_text:
        if 'a' <= character <= 'z':  # 小写字母
            plain_text += chr(ord('a') + (ord(character) - ord('a') - alphabet_to_unicode(key[key_index])) % 26)
        elif 'A' <= character <= 'Z':  # 大写字母
            plain_text += chr(ord('A') + (ord(character) - ord('A') - alphabet_to_unicode(key[key_index])) % 26)
        else:  # 其他字符不处理
            plain_text += character
        key_index = next_index(key_index, len(key))  # 下一个密钥序号
    return plain_text


word = input("输入要处理的改进的维吉尼亚密码的密/明文：")
keys = input("输入维吉尼亚密码密钥：")
print("-----用" + keys + "对" + word + "加密后是：-----\n" + encryption(word, keys))
print("-----用" + keys + "对" + word + "解密后是：-----\n" + decryption(word, keys))
print("-----验证-----")
print("-----用" + keys + "对" + decryption(word, keys) + "加密后是：-----\n" + encryption(decryption(word, keys), keys))
print("-----用" + keys + "对" + encryption(word, keys) + "解密后是：-----\n" + decryption(encryption(word, keys), keys))
