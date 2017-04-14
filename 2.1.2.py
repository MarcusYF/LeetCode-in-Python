# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 21:58:23 2016

@author: yaofan29597
"""

#%% remove duplicates from sorted array(with multiple occurances)

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