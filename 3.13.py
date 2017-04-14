# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 00:11:12 2016

@author: yaofan29597
"""

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

