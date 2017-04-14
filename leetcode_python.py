# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:10:18 2017

@author: yaofan29597
"""

#%% 2.1.1 remove duplicates from sorted array 

import pandas as pd

A = pd.Series(range(0, 10))
A = pd.DataFrame.sample(A, 20, replace=True)
A = A.get_values()
A.sort()

class solution:
    def __init__(self, A):
        self.A = []
        self.A[:] = A
    def removeDuplicates(self):
        n = len(self.A)
        if n == 0:
            return 0
        index = 0
        for i in range(1, n):
            if self.A[index] != self.A[i]:
                index += 1
                self.A[index] = self.A[i]
        self.A = self.A[:index+1]
        return index+1
        
test = solution(A)
test.removeDuplicates()

#%% 2.1.2 remove duplicates from sorted array(with multiple occurances)

import pandas as pd

A = pd.Series(range(0, 10))
A = pd.DataFrame.sample(A, 20, replace=True)
A = A.get_values()
A.sort()

class solution:
    def __init__(self, A):
        self.A = []
        self.A[:] = A
    def removeDuplicates(self, occur_allowed):
        n = len(self.A)
        if n <= occur_allowed:
            return n
        index = occur_allowed
        for i in range(occur_allowed, n):
            if self.A[index-occur_allowed] != self.A[i]:
                self.A[index] = self.A[i]
                index += 1
        self.A = self.A[:index]
        return index
        
test = solution(A)
test.removeDuplicates(1)

#%% 2.1.3 search in rotated sorted array

import random
A = list(range(20))
target = random.randint(0,19)
A[:] = A[target:] + A[:target]

class solution:
    def __init__(self, A):
        self.A = []
        self.A[:] = A
    def search(self, target):
        first = 0
        last = len(self.A)
        while first != last:
            mid = int((first + last) / 2)
            if self.A[mid] == target:
                return mid
            if self.A[first] <= self.A[mid]:
                if self.A[first] <= target and target < self.A[mid]:
                    last = mid
                else:
                    first = mid + 1
            else:
                if self.A[mid] < target and target <= self.A[last-1]:
                    first = mid + 1
                else:
                    last = mid
        return -1

test = solution(A)
test.search(4)

#%% 2.1.5 median of two sorted arrays

import random
m = 20; n = 11
A = random.sample(range(50), m); A = sorted(A)
B = random.sample(range(20), n); B = sorted(B)

class solution:
    def findMedianSortedArrays(self, A, m, B, n):
        tot = m + n
        if tot % 2 != 0:
            return self.find_kth(A, m, B, n, int((tot+1)/2))
        else:
            return (self.find_kth(A, m, B, n, int(tot/2)+1) + self.find_kth(A, m, B, n, int(tot/2)))/2

    def find_kth(self, A, m, B, n, k):
        if m > n:
            return self.find_kth(B, n, A, m, k)
        if m == 0:
            return B[k-1]
        if k == 1:
            return min(A[0], B[0])
        pa = min(int(k/2), m); pb = k - pa
        if A[pa - 1] < B[pb - 1]:
            return self.find_kth(A[pa:], m - pa, B, n, k - pa)
        elif A[pa - 1] > B[pb - 1]:
            return self.find_kth(A, m, B[pb:], n - pb, k - pb)
        else:
            return A[pa - 1]
            
test = solution()
test.findMedianSortedArrays(A, m, B, n)

#%% 2.1.6 longest consecutive sequence

import random
A = random.sample(range(1,50), 30)
A_map = {}.fromkeys(A, 0)

class solution:
    def longestConsecutive(self, A):
        A_map = {}.fromkeys(A, 0)
        n = max(A)
        longest = 0
        for key in A:
            length = 1
            if A_map[key] != 0:
                continue
            for i in range(key + 1, n + 1):
                if not A_map.get(i, True) :
                    length += 1
                    A_map[i] = 1
                else:
                    break
            start = key
            for i in range(key - 1, -1, -1):       
                if not A_map.get(i, True):
                    length += 1
                    start -= 1
                    A_map[i] = 1
                else:
                    break
            if length > longest:
                longest = length
                start_position = start
        return longest, start_position

test = solution()
test.longestConsecutive(A)

#%% 2.1.7 two sum

import random

class solution:
    def find_2_sum_sorted(self, A, n, s):
        res = []
        A.sort()
        i = 0; e = n - 1
        while (i < e):
            tmp = A[i] + A[e]
            if tmp > s:
                e -= 1
            elif tmp < s:
                i += 1
            else:
                res.append((A[i], A[e]))
                i += 1; e -= 1
        return res
    
    def find_2_sum(self, A, n, s):
        res = []
        B = set(A)
        for item in A:
            if (item <= (s>>1)) and s - item in B:
                res.append((item, s - item))
        return res
                

test = solution()

r = 100; l = 50;
A = random.sample(range(r), l)
%timeit [test.find_2_sum(A, l, r)]
%timeit [test.find_2_sum_sorted(A, l, r)]
       
#%% three sum

class Solution:
    # @return a list of lists of length 3, [[val1,val2,val3]]
    def threeSum(self, num):
        num.sort()
        dct, ans = {}, []
        for i in range(0, len(num)):
            if (i > 0 and num[i] == num[i-1]):
                continue
            l, r = i + 1, len(num) - 1
            while l < r:
                sum = num[l] + num[r] + num[i]
                if sum == 0:
                    ans.append([num[i], num[l], num[r]])
                    while l < r and num[l] == num[l + 1]: l = l + 1
                    while l < r and num[r] == num[r - 1]: r = r - 1
                    l, r = l + 1, r - 1
                elif sum < 0:
                    l = l + 1
                else:
                    r = r - 1   
        return ans    

t = Solution()
t.threeSum([-1, 2, 3, -5])


#%% 2.1.12 next permutation

import random
A = random.sample(range(10), 10)
A = [1,2,3]

class solution:
    def swap(self, a, p, q):
        a[p], a[q] = a[q], a[p]
        
    def find_first_larger(self, a, x, left, right):
        if left == right:
            return left
        p = int((left + right) / 2)
        if x > a[p]:
            return self.find_first_larger(a, x, left, p-1)
        elif x < a[p]:
            return self.find_first_larger(a, x, p, right)
        else:
            return p
        
    def nextPermutation(self, A):
        n = len(A)
        index = -1
        for i in range(-2, -n-1, -1):
            if A[i] < A[i+1]:
                index = i+1
                break
            index = -n
        
        if index == -n:
            # copy
            A[:] = A[::-1]
            return A
        else:
            pivot = self.find_first_larger(A[index:], A[index-1], index, -1)
            
        self.swap(A, pivot, index-1)
        A[index:] = A[:n-1+index:-1]
        
        return A

test = solution()
test.nextPermutation(A)

#%% 2.1.15 Trapping Rain Water

import numpy as np
A = np.random.geometric(0.5, 100) - 1

class solution:
    def trap(self, A):
        n = len(A)
        highest = 0
        for index, value in enumerate(A):
            if value > highest:
                highest = value
                highest_index = index
                
        water = 0
        peak0 = 0
        for index, value in enumerate(A[:highest_index]):
            if value > peak0:
                peak0 = value
            else:
                water += peak0 - value
        peak1 = 0
        for index, value in enumerate(A[:highest_index-n:-1]):
            if value > peak1:
                peak1 = value
            else:
                water += peak1 - value
        return water
        
test = solution()
test.trap(A)

#%% 2.1.16 Rotate Image

import numpy as np
N = 9
A = np.random.random(size=(N, N)) * 10
A = np.round(A).astype(int)

class solution:
    
    def swap(self, A, p, q, r, s):
        A[p, q], A[r, s] = A[r, s], A[p, q]
    
    def reflect_1001(self, A):
        n = len(A)
        for i in range(n-1):
            for j in range(n-1-i):
                self.swap(A, i, j, -j-1, -i-1)
    
    def reflect_0001(self, A):
        n = len(A)
        for i in range(int(n/2)):
            for j in range(n):
                self.swap(A, i, j, -i-1, j)
        
    def rotate(self, A):
        self.reflect_1001(A)
        self.reflect_0001(A)

test = solution()
test.rotate(A)

#%% 2.1.23 find single number from twice

import numpy as np
import random

A = random.sample(range(10),6)*2
A.pop(0)
np.random.shuffle(A)

class solution:
    def popSingleNumber(self, A):
        x = 0
        for n in A:
            x ^= n
        return x

test = solution()
test.popSingleNumber(A)

#%% 3.13 find anagrams

A = ['abs', 'asb', 'ac']

filename = '/Users/yaofan29597/Desktop/Princetechs/学习资料/pyspark测试/composition.txt'
fr = open(filename)
A = []
for line in fr.readlines():
    line = line.strip()
    A += line.split(' ')

class solution:
    def sort_str(self, word):
        return ''.join(sorted(word))
    def anagrams(self, A):
        v = list(map(self.sort_str, A))
        B = dict(zip(A, v))
        return sorted(B.items(), key=lambda d:d[1])

test = solution()
B = test.anagrams(A)

#%% 6.6 find first missing positive

from random import sample
A = sample(range(-10,10),15)
#A[i] == A[A[i]-1]
class solution:
    def swap(self, A, p, q):
        A[p], A[q] = A[q], A[p]
        
    def bucket_sort(self, A):
        n = len(A)
        for i in range(n):
            while(A[i] != i+1):
                if A[i]<=0 or A[i]>n-1 or A[i]==i+1:
                    break
                else:
                    self.swap(A, i, A[i]-1)
        return A

test = solution()
test.bucket_sort(A)

#%% enumerate distinct subsets in S

class Solution:
    # @param num, a list of integer
    # @return a list of lists of integer
    def subsetsWithDup(self, S):
        S.sort()
        bset = []
        for x in range(2**len(S)):
            for i in range(1, len(S)):
                if (S[i] == S[i-1] and (x>>(i-1)&0x03 == 0x01)): break
            else:
                bset.append(x)
        return [[S[x] for x in range(len(S)) if i>>x&1] for i in bset]

#%% graycode

class Solution:
    # @return a list of integers
    def grayCode(self, n):
        self.res = [0]
        for i in [2**x for x in range(0, n)]:
            self.res.append(self.res[-1] + i)
            self.res.extend([i + v for v in self.res[-3:None:-1]])
        return self.res

#%% merge two sorted array

class Solution:
    # @param A  a list of integers
    # @param m  an integer, length of A
    # @param B  a list of integers
    # @param n  an integer, length of B
    # @return nothing
    def merge(self, A, m, B, n):
        for i in range(m + n - 1, -1, -1):
            if m == 0 or (n > 0 and B[n-1] > A[m-1]):
                A[i] = B[n-1]
                n -= 1
            else:
                A[i] = A[m-1]
                m -= 1
        return A

#%% Hamming distance

class Solution:
    # @return an integer
    def minDistance(self, word1, word2):
        dp = [[0] * (len(word2) + 1) for i in range(len(word1) + 1)]
        for i in range(1, len(word1) + 1): 
            dp[i][0] = i
        for i in range(1, len(word2) + 1): 
            dp[0][i] = i
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                dp[i][j] = dp[i - 1][j - 1] + 1
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1)
                dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1)
        return dp[len(word1)][len(word2)]
        
#%% quick sort
l = [5,1,3,8,9,4,7,6]

class Solution:
    def path_sort(self, l, start_index, end_index):
        flag = l[end_index]
        i = start_index - 1
        for j in range(start_index,end_index):
            if l[j] > flag:
                pass
            else:
                i += 1
                l[i], l[j] = l[j], l[i]
        l[end_index], l[i+1] = l[i+1], l[end_index]

        return i+1

    def Quick_sort(self, l, start_index, end_index):
        if start_index >= end_index:
            return
        middle = self.path_sort(l, start_index, end_index)
        self.Quick_sort(l, start_index, middle-1)
        self.Quick_sort(l, middle + 1, end_index)


        
t = Solution()
t.Quick_sort(l,0,len(l)-1)

#%% combination sum

class Solution(object):
    def find(self, k, n, start):
        if k > 1:
            res = []
            upper_bound = min((2 * n - k * k + k) // (2 * k) + 1, 10)
            
            if upper_bound > start:
                flag = 0
                for i in range(start, upper_bound):
                    subList = self.find(k - 1, n - i, i + 1)
                    if subList:
                        res.extend(list(map(lambda x:[i]+x, subList)))
                        flag += 1
                if flag == 0:
                    return []
            else:
                return []
            return res
        else:
            if start <= n < 10:
                return [[n]]
            else:
                return []
            
    def combinationSum3(self, k, n):

        return self.find(k, n, 1)
        
#%% find the duplicate number
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        fast = n
        slow = n
        while True:
            fast = nums[nums[fast-1]-1]
            slow = nums[slow-1]
            if fast == slow:
                break
        fast = n
        while (slow != fast):
            slow = nums[slow-1]
            fast = nums[fast-1]
        return slow
        
#%% combination sum(DFS)
class Solution(object):
    def combinationSum(self, candidates, target):
        res = []
        candidates.sort()
        self.dfs(candidates, target, 0, [], res)
        return res
    
    def dfs(self, nums, target, index, path, res):
        if target < 0:
            return  # backtracking
        if target == 0:
            res.append(path)
            return 
        for i in xrange(index, len(nums)):
            self.dfs(nums, target-nums[i], i, path+[nums[i]], res)
            
#%% word search(DFS)
class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        self.m = len(board)
        self.n = len(board[0])
        
        for x in range(self.m):
            for y in range(self.n):
                if self.exist_recur(board, x, y, word, 0):
                    return True
        return False
        
    def exist_recur(self, board, x, y, word, i):
        if i > len(word) - 1:
            return True
        if x < 0 or x > self.m - 1 or y < 0 or y > self.n - 1 or board[x][y] != word[i]:
            return False
        tmp = board[x][y]
        board[x][y] = '#' # 精妙
        flag = self.exist_recur(board, x+1, y, word, i+1) or \
               self.exist_recur(board, x-1, y, word, i+1) or \
               self.exist_recur(board, x, y+1, word, i+1) or \
               self.exist_recur(board, x, y-1, word, i+1)
        board[x][y] = tmp
        return flag
        

#%% binary search in matrix
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        row, isFind = self.searchColumn(matrix, target)
        if isFind:
            return True
        else:
            return self.searchRow(matrix[row], target)
    
    def searchColumn(self, matrix, target):
        l = 0
        r = len(matrix) - 1
        while r > l + 1:
            m = int((r + l) / 2)
            if matrix[m][0] < target:
                l = m
            elif matrix[m][0] > target:
                r = m - 1
            else:
                return m, True
        if target == matrix[r][0]:
            return r, True
        elif target < matrix[r][0]:
            return l, False
        else:
            return r, False
        
    def searchRow(self, row, target):
        l = 0
        r = len(row) - 1
        while r > l:
            m = int((r + l) / 2)
            if row[m] < target:
                l = m + 1
            elif row[m] > target:
                r = m - 1
            else:
                return True
        if target == row[l]:
            return True
        else:
            return False
        
        

import numpy as np

class Solution(object):
    def constructRectangle(self, area):
        """
        :type area: int
        :rtype: List[int]
        """
        L = area
        W = 1
        n = int(np.sqrt(area))
        for W in range(n, 1, -1):
            print(W)
            if area % W == 0:
                L = int(area / W)
                print(L, W)
                return [L, W]
        return [L, W]
t=Solution()

            



