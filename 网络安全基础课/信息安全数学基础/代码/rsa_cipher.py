# 编程实现2^200位密钥的RSA公钥密码系统
import random

from euclid_inverse import recursive_euclid  # 导入最大公因数
from euclid_inverse import inverse  # 导入求逆
from three_x_plus_1_and_goldbach import prime_array_gen  # 导入生成素数数组


def key_gen(up_limit=pow(2, 100), button_limit=2):
    print("开始构建素数集，上限为{0}，下限为{1}".format(up_limit, button_limit))
    primes = prime_array_gen(up_limit, button_limit)
    print("构建素数集完成，上限为{0}，下限为{1}".format(up_limit, button_limit))
    print(primes)
    p = random.choice(primes)
    q = random.choice(primes)
    print("随机抽取素数p、q为", p, q)
    n = p * q
    phi_n = (p - 1) * (q - 1)
    e = random.choice(range(2, phi_n))
    while recursive_euclid(e, phi_n) != 1:
        e = random.choice(range(2, phi_n))
    d = inverse(phi_n, e)
    print("公钥为", e, n, "私钥为", d, n)
    return e, n, d


def encryption(plain_num, e, n):  # 加密
    secret_num = pow(plain_num, e) % n
    return secret_num


def decryption(secret_num, d, n):  # 解密
    plain_num = pow(secret_num, d) % n
    return plain_num


if __name__ == '__main__':
    e, n, d = key_gen(100000)
    text = input("请输入要用于RSA加密的数字（要小于" + str(n) + "）")
    print("=====对" + text + "加密=====")
    encrypted = encryption(int(text), e, n)
    print("加密结果为：" + str(encrypted))
    print("=====对" + str(encrypted) + "解密=====")
    decrypted = decryption(encrypted, d, n)
    print("解密结果为：" + str(decrypted))
