# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 20:16:31 2016

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

    def remov_duplicates(self):
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