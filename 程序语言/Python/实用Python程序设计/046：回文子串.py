a = input()
for i in range(2, len(a) + 1):
    for j in range(len(a)):
        if i + j <= len(a):
            pos = a[j:j + i]
            if j == 0:
                neg = a[j + i - 1::-1]
            else:
                neg = a[j + i - 1:j - 1:-1]
            if pos == neg:
                print(pos)
