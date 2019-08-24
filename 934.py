import collections

class Solution():

    def find_island(self, r, c):
        if 0 <= r < len(self.A) and 0 <= c < len(self.A) and self.A[r][c] == 1:
            self.A[r][c] = -1
            self.find_island(r + 1, c)
            self.find_island(r - 1, c)
            self.find_island(r, c + 1)
            self.find_island(r, c - 1)

    def split_island(self, A):
        self.A = A
        for r, row in enumerate(A):
            for c, v in enumerate(row):
                if v == 1:
                    self.find_island(r, c)
                    return

    def shortestBridge(self, A):
        self.split_island(A)

        flag = True
        edge = 0
        while flag:
            edge += 1
            for r, row in enumerate(A):
                for c, v in enumerate(row):
                    if v == edge:
                        if r+1 < len(self.A) and A[r+1][c]==-1 or r-1 >= 0 and A[r-1][c]==-1 or c+1 < len(self.A) and A[r][c+1]==-1 or c-1 >= 0 and A[r][c-1]==-1:
                            flag = False
                        if r+1 < len(self.A) and A[r+1][c]==0:
                            A[r + 1][c] = edge + 1
                        if r-1 >= 0 and A[r-1][c]==0:
                            A[r - 1][c] = edge + 1
                        if c+1 < len(self.A) and A[r][c+1]==0:
                            A[r][c + 1] = edge + 1
                        if c-1 >= 0 and A[r][c-1]==0:
                            A[r][c - 1] = edge + 1

        return edge-1



A = [[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]
# A = [[0,1,0],[0,0,0],[0,0,1]]

s = Solution()
res = s.shortestBridge(A)