# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 00:00:08 2016

@author: yaofan29597
"""

#%% next permutation

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
        
        
        
        
        
        
                
        
        
