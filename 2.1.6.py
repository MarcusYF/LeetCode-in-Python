# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 23:09:35 2016

@author: yaofan29597
"""

#%% longest consecutive sequence

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
        
            
                
                    
        