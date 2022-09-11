# Find the missing number in a limited range list `arr[1…n+1]`
def findMissingElement(arr):
 
    n = len(arr)
 
    # calculate the sum of all elements of input list
    total = sum(arr)
 
    # expected sum - actual sum
    return  (n + 1) * (n + 2) // 2 - total
 
 
if __name__ == '__main__':
 
    # input list contains `n` numbers between 1 and `n+1`
    # with one number missing and no duplicates
    arr = [3, 2, 4, 6, 1]
 
    print('The missing element is', findMissingElement(arr))