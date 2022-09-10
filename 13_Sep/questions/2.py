# Function to remove adjacent duplicates characters from a string
def removeDuplicates(s):
    chars = []
    prev = None
 
    for c in s:
        if prev != c:
            chars.append(c)
            prev = c
 
    return ''.join(chars)
 
 
if __name__ == '__main__':
 
    s = 'AAABBCDDD'
    print(removeDuplicates(s))