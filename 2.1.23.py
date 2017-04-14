# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:51:29 2016

@author: yaofan29597
"""

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
            