# 040:万年历
def is_leap_year(year):  # 判断是否是闰年
    if year % 4 == 0 and year % 100 or year % 400 == 0:
        return True
    else:
        return False


def calculate(string):  # 计算周几
    weekdays = "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday"  # 已知的为周三
    y1, m1, d1 = int(string[0]), int(string[1]), int(string[2])  # 待求日期
    y2, m2, d2 = 2020, 11, 18  # 已知日期
    # print('day%7',days(y1, m1, d1, y2, m2, d2) % 7)
    return weekdays[days(y1, m1, d1, y2, m2, d2) % 7]


def days(year1, month1, day1, year2, month2, day2):  # 计算和已知的天数差距，1：待求2：已知
    day = 0
    # 加上两个年份带来的的天数差距（多加少减）
    if year1 >= year2:  # 多加
        day += calculate_day_year(year2, year1)
    else:  # 少减
        day -= calculate_day_year(year1, year2)
    day += calculate_day_month(year1, month1) - calculate_day_month(year2, month2)  # 加上两个月份带来的的天数差距
    day += day1 - day2  # 加上两个日期带来的的天数差距
    # print("day",day)
    return day


def calculate_day_year(year_min, year_max):  # 计算两年份的天数差距
    day_y = 0
    for y in range(year_min, year_max):  # 从小到大遍历年份
        if is_leap_year(y):  # 是闰年？
            day_y += 366
        else:  # 不是闰年
            day_y += 365
    # print("day_y", day)
    return day_y


def calculate_day_month(year, month):  # 计算从一年开头到这个月一日前的天数
    day_m = 0
    month_days_leap = (0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)  # 闰年的月天数元组
    month_days = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)  # 平年的月天数元组
    if is_leap_year(year):  # 是闰年？
        for m in range(month):  # 闰年遍历月天数元组
            day_m += month_days_leap[i]
    else:  # 不是闰年
        for m in range(month):  # 平年年遍历月天数元组
            day_m += month_days[i]
    # print("day_m", day)
    return day_m


for i in range(int(input())):  # 有这么多组日期要计算
    strings = input().split()  # 拆分年月日
    if int(strings[1]) in range(1, 13):  # 月合法？
        if strings[1] == "2":  # 是2月
            if is_leap_year(int(strings[0])):  # 是闰年？
                if int(strings[2]) in range(1, 30):  # 闰年2月日期合法？
                    print(calculate(strings))  # 求解
                else:  # 不合法
                    print('Illegal')
            else:  # 不是闰年
                if int(strings[2]) in range(1, 29):  # 平年2月日期合法？
                    print(calculate(strings))  # 求解
                else:  # 不合法
                    print('Illegal')
        elif strings[1] in ("1", "3", "5", "7", "8", "10", "12"):  # 不是二月，是大月？
            if int(strings[2]) in range(1, 32):  # 大月日期合法？
                print(calculate(strings))  # 求解
            else:  # 不合法
                print('Illegal')
        else:  # 是小月
            if int(strings[2]) in range(1, 31):  # 小月日期合法？
                print(calculate(strings))  # 求解
            else:  # 不合法
                print('Illegal')
    else:  # 不合法
        print('Illegal')
