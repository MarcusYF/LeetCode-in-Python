from collections import deque

# Definition for a Node.
class Node:
    def __init__(self, val, children):
        self.val = val
        self.children = children

class Solution:
    def levelOrder(self, root: 'Node'):
        if root is None:
            return []
        q = deque([(0, root)])
        res = []
        while q:
            l, n = q.popleft()
            if len(res) < l + 1:
                res.append([])
            res[l].append(n.val)
            for c in n.children:
                q.append([l+1, c])
        return res

s = Solution()
c = Node(7, [])
d = Node(8, [])
e = Node(9, [])
f = Node(10, [])
b = Node(2, [e,f])
a = Node(3, [c,d])
tree = Node(1, [a,b])
r = s.levelOrder(tree)
