def encryption(plain_text, letters_per_round):  # 加密
    if len(plain_text) <= letters_per_round:
        return "无效的，因为纸条缠不超过一圈"
    cipher_text = ""  # 初始化密文
    complete_rounds = len(plain_text) // letters_per_round
    remainder = len(plain_text) % letters_per_round  # 考虑最后一圈有几个多出来的
    # print("complete_rounds", complete_rounds, remainder)
    for current_round in range(complete_rounds):  # 处理前面完整的complete_rounds圈
        current_index = current_round
        # print(" current_round", current_index)
        remainder_jump = 0  #
        while current_index < len(plain_text):
            # print("  current_index", current_index)
            cipher_text = cipher_text + plain_text[current_index]
            if remainder_jump < remainder:
                current_index += 1
                remainder_jump += 1
            current_index += complete_rounds
    else:  # 处理最后一列多出的remainder个
        if remainder:
            current_index = complete_rounds
            for current_remainder in range(remainder):
                cipher_text += plain_text[current_index]
                current_index += complete_rounds + 1
    return cipher_text


def decryption(cipher_text, letters_per_round):  # 解密
    if len(cipher_text) <= letters_per_round:
        return "无效的，因为纸条缠不超过一圈"
    plain_text = ""  # 初始化密文
    for current_round in range(letters_per_round):
        current_index = current_round
        while current_index < len(cipher_text):
            plain_text += cipher_text[current_index]
            current_index += letters_per_round
    return plain_text


word = input("输入要处理的斯巴达木卷的密/明文：")
maxI = int(input("输入最大每圈可能有几个字母："))
# print(len(word))
print("-----将" + word + "作为明文加密-----")
for i in range(maxI):
    print("每圈" + str(i + 1) + "个字母的话，对其加密后是：" + encryption(word, i + 1))
print("-----将" + word + "作为密文尝试解密-----")
for i in range(maxI):
    print("每圈" + str(i + 1) + "个字母的话，对其解密后是：" + decryption(word, i + 1))
