# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 23:00:30 2016

@author: yaofan29597
"""

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
        
            