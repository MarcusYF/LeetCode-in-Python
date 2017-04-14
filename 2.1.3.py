# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 14:35:47 2016

@author: yaofan29597
"""

#%% search in rotated sorted array

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