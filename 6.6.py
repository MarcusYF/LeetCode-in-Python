# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 23:36:34 2016

@author: yaofan29597
"""

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