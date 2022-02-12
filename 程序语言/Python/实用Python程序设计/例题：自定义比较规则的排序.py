students = [
    ('John', 'a', 15),
    ('Mike', 'b', 12),
    ('Mike', 'c', 18),
    ('Bom', 'd', 10)
]
students.sort()  # 按元素大小排列（元组比大小：依次按照元组内各元素排序）
print(students)
students.sort(key=lambda x: x[2])  # 按第三个元素排序
print(sorted(students, key=lambda x: x[0], reverse=True))  # 按第二个元素逆序排序并打印
