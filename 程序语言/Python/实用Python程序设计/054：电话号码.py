import re


def get_phone_num(what_in_tag):
    flag = True
    num = re.findall('(\\([0-9]{1,2}\\)-[0-9]{3})', what_in_tag[1])  # 可能是电话号码的片段
    not_num = re.findall('(\\([0-9]{1,2}\\)-[0-9]{3})[0-9]', what_in_tag[1])  # 需要排除的片段
    # 从num中剔除和not_num一样的项目
    for num_id in range(len(num)):
        for not_num_id in range(len(not_num)):
            if num[num_id] == not_num[not_num_id]:
                num[num_id] = ''
    # print(what_in_tag[0], num)
    for num_id in range(len(num)):
        num[num_id] = re.findall('\\(([0-9]{1,2})\\)', num[num_id])
    # print(what_in_tag[0], num)
    if num:
        if num[0]:
            # print(len(num))
            print('<' + what_in_tag[0] + '>', end="")
            for num_id in range(len(num) - 1):
                print(num[num_id][0] + ',', end='')
            print(num[len(num) - 1][0], end='')
            print('</' + what_in_tag[0] + '>')
            flag = False
    # print('flag=',flag)
    return flag


for i in range(int(input())):
    sayNone = True
    sample = input()
    whatInTag = re.findall('<([a-z]+?)>(.*?)</\\1>', sample)
    # print(whatInTag)
    for j in whatInTag:
        # print('sayNoneF=', sayNone)
        sayNone = get_phone_num(j) and sayNone
        # print('sayNoneA=',sayNone)
    whatInTag = []
    if sayNone:
        print('NONE')
