def encryption(plain_text, key):  # 右移加密
    cipher_text = ""  # 初始化密文
    for character in plain_text:
        if 'a' <= character <= 'z':
            cipher_text += chr(ord('a') + (ord(character) - ord('a') + key) % 26)
        elif 'A' <= character <= 'Z':
            cipher_text += chr(ord('A') + (ord(character) - ord('A') + key) % 26)
        else:
            cipher_text += character
    return cipher_text


def decryption(cipher_text, key):  # 左移解密
    plain_text = ""  # 初始化明文
    for character in cipher_text:
        if 'a' <= character <= 'z':
            plain_text += chr(ord('a') + (ord(character) - ord('a') - key) % 26)
        elif 'A' <= character <= 'Z':
            plain_text += chr(ord('A') + (ord(character) - ord('A') - key) % 26)
        else:
            plain_text += character
    return plain_text


word = input("输入要处理的改进的凯撒密码的密/明文：")
print("-----将" + word + "作为明文加密-----")
for i in range(26):
    print("加密密钥为" + str(i + 1) + "的话，对其加密后是：" + encryption(word, i + 1))
print("-----将" + word + "作为密文尝试解密-----")
for i in range(26):
    print("加密密钥为" + str(i + 1) + "的话，对其解密后是：" + decryption(word, i + 1))
