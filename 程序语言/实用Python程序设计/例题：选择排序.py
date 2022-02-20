def selection_sort(a):
    n = len(a)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if a[i] > a[j]:
                a[i], a[j] = a[j], a[i]


lst = [1, 12, 4, 56, 6, 2]
selection_sort(lst)
print(lst)
