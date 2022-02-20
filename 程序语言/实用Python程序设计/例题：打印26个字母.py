# 打印26个字母
for i in range(26):
    print(chr(ord("a") + i), end="")  # ord(x)求字符x的编码（ASCII编码），chr（x）就编码为x的字符
