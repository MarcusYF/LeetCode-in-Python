class Solution(object):

    def countBattleships(self, board):
        """
        :type board: List[List[str]]
        :rtype: int
        """
        m = len(board)
        if 0 == m: return 0
        n = len(board[0])
        count = 0

        for i in range(m):
            for j in range(n):
                if '.' == board[i][j]: continue
                if i > 0 and 'X' == board[i - 1][j]: continue
                if j > 0 and 'X' == board[i][j - 1]: continue
                count += 1

        return count