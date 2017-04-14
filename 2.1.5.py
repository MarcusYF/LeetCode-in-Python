# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 23:48:00 2016

@author: yaofan29597
"""

#%% median of two sorted arrays

import random
m = 20; n = 11
A = random.sample(range(50), m); A = sorted(A)
B = random.sample(range(20), n); B = sorted(B)

class solution:
    def findMedianSortedArrays(self, A, m, B, n):
        tot = m + n
        if tot % 2 != 0:
            return self.find_kth(A, m, B, n, int((tot+1)/2))
        else:
            return (self.find_kth(A, m, B, n, int(tot/2)+1) + self.find_kth(A, m, B, n, int(tot/2)))/2

    def find_kth(self, A, m, B, n, k):
        if m > n:
            return self.find_kth(B, n, A, m, k)
        if m == 0:
            return B[k-1]
        if k == 1:
            return min(A[0], B[0])
        pa = min(int(k/2), m); pb = k - pa
        if A[pa - 1] < B[pb - 1]:
            return self.find_kth(A[pa:], m - pa, B, n, k - pa)
        elif A[pa - 1] > B[pb - 1]:
            return self.find_kth(A, m, B[pb:], n - pb, k - pb)
        else:
            return A[pa - 1]
            
test = solution()
test.findMedianSortedArrays(A, m, B, n)

C = sorted(A+B)
if (m + n) % 2 != 0:
    median = C[int((m+n-1)/2)]
else:
    median = (C[int((m+n)/2)] + C[int((m+n)/2)-1])/2
        
        
            