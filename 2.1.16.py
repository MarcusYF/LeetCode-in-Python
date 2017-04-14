# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:00:20 2016

@author: yaofan29597
"""

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
        