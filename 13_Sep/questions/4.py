
def swap(a, i, j):
 
    temp = a[i]
    a[i] = a[j]
    a[j] = temp
 
 
def partition(a):
 
    pIndex = 0
 
    # each time we find a negative number, `pIndex` is incremented,
    # and that element would be placed before the pivot
    for i in range(len(a)):
        if a[i] < 0:        # pivot is 0
            swap(a, i, pIndex)
            pIndex = pIndex + 1
 
 
if __name__ == '__main__':
 
    a = [9, -3, 5, -2, -8, -6, 1, 3]
    partition(a)
    print(a)