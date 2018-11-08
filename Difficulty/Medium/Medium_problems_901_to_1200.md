# Problem 901~1200 Medium

## 901. Online Stock Span

**题意**：

## 904. Fruit Into Baskets

**题意**：

## 907. Sum of Subarray Minimums

**题意**：

## 909. Snakes and Ladders

**题意**：

## 910. Smallest Range II

**题意**：

## 911. Online Election

**题意**：

## 915. Partition Array into Disjoint Intervals

**题意**：

## 916. Word Subsets

**题意**：

## 918. Maximum Sum Circular Subarray

**题意**：

## 919. Complete Binary Tree Inserter

**题意**：

## 921. Minimum Add to Make Parentheses Valid

**题意**：


## 923. 3Sum With Multiplicity

**题意**：给你一个

**思路**：

代码如下：

    class Solution {
    public:
        int threeSumMulti(vector<int>& A, int target) {
            int n = A.size();
            unordered_map<int, int> valhash, sumhash;
            long long ans = 0;
            const int mod = 1e9 + 7;
            for(int i = 0; i < n; i++)
            {
                for(auto it = sumhash.begin(); it != sumhash.end(); it++)
                {
                    int rem = target - it -> first, cnt = it -> second;
                    if(rem == A[i])
                        ans = (ans + cnt) % mod;
                }
                for(auto it = valhash.begin(); it != valhash.end(); it++)
                {
                    int sum = A[i] + it -> first, cnt = it -> second;
                    if(sum <= target)
                        sumhash[sum] += cnt;
                }
                valhash[A[i]]++;
            }
            return ans;
        }
    };

## 926. Flip String to Monotone Increasing 

**题意**：

## 930. Binary Subarrays With Sum

**题意**：给你一个只包含`0`和`1`的数组`A`，问你A中有多少个**非空子数组**的和是`S`？

例子：

**Input**: A = [1,0,1,0,1], S = 2  
**Output**: 4  
**Explanation**:  
The 4 subarrays are bolded below:  
[1,0,1,0,1]  
[1,0,1,0,1]  
[1,0,1,0,1]  
[1,0,1,0,1]  

数据范围：

1. `A.length <= 30000`
2. `0 <= S <= A.length`
3. `A[i] is either 0 or 1`

**思路**：前缀和和hash，很容易想到。

代码如下：

    class Solution {
    public:
        int numSubarraysWithSum(vector<int>& A, int S) {
            int n = A.size();
            vector<int> sum(n + 1, 0);
            unordered_map<int, int> umap;
            umap[0]++;
            int ans = 0;
            for(int i = 1; i <= n; i++)
            {
                sum[i] = sum[i - 1] + A[i - 1];
                int pre = sum[i] - S;
                ans += umap[pre];
                umap[sum[i]]++;
            }
            return ans;
        }
    };


## 931. Minimum Falling Path Sum

题意：给你一个整数的方阵A，让你最小化A的`falling path`的sum和。

一个`falling path`是从第一行的任意一个数字开始，然后往下，每次选择下一行的列时，列号只能在当前这个位置的列号基础上变化`[-1, 1]`（相当于最多有三种转移方式）。

思路：水dp，没啥说的。

代码如下：

    class Solution {
    public:
        int minFallingPathSum(vector<vector<int>>& A) {
            int n = A.size(), m = A[0].size();
            for(int i = 1; i < n; i++)
            {
                for(int j = 0; j < m; j++)
                {
                    int res = 1e9;
                    for(int k = -1; k <= 1; k++)
                    {
                        if(j + k >= 0 && j + k < m)
                        {
                            res = min(res, A[i - 1][j + k]);
                        }
                    }
                    A[i][j] += res;
                }
            }
            int ans = 1e9;
            for(int j = 0; j < m; j++) ans = min(ans, A[n - 1][j]);
            return ans;
        }
    };

## 932. Beautiful Array

**题意**：一个长度为N的整数数组，其中的数字是`1,2,3，..., N`。如果这个数组对于任意的 `0 <= i < k < j < N`, 都有`A[k] * 2 != A[i] + A[j]`，则这个数组是beautiful的。

给你一个N(1 <= N <= 1000)，让你构造任意一个这样的beautiful序列。

**思路**：这道题确实没太想出来。主要是受到了别的类似的题的影响，注意在思考问题的时候不要刻意去思考目前的这道题在之前有没有类似的题，题与题之间的差异往往很大，借鉴性不强。一定要注意**具体问题具体分析**。之前的题目([HDU 3833](http://acm.hdu.edu.cn/showproblem.php?pid=3833))是N很大(1 <= N <= 10000)，判断数组中是否存在`0 <= i < k < j`满足`A[k] * 2 == A[i] + A[j]`。使用的是hash方法，看每个数字放置的先后顺序，以及小于这个数字的区间和大于这个数字的相同长度区间对应位置的数是不是一个已经放置了，而一个未放置。如果两侧的数组是对称的，则`A[k] * 2 != A[i] + A[j]`，但是使用这种方法来构造会很难。但是判断`A[k] * 2 == A[i] + A[j]`却比较简单。因此不要刻意去把别的题的方法用在目前做的题上。注意**具体问题具体分析**，忘记自己之前做过的题。

因此**要分析beautiful序列的规律和特点，这个题目不管怎么变，都是有效的**。

在分析特点的时候，考察序列是否满足整体性和部分性这个是很重要的。把大问题转化成小的问题，这种思想在计算机中应用太多，对于解决问题而言也是很重要的。

可以知道，`[1]`是beautiful的， `[1,2]`也是beautiful的, `[1,3,2]`也是beautiful array。

一个重要的规律是， 如果一个数组是beautiful的， 那么任意挑选其中的元素， 按照原来的顺序组成数组，也是beautiful array。

这个比较容易理解， 整个数组里面挑选不出`A[k]* 2 = A[i] + A[j]`的话， 那其中一部分也一定挑选不出来。

这时候，如果我们可以构造一个大一点的数组， 那么把其中<=N的数挑选出来，就可以返回一个符合要求的结果了。因此就可以把整体的问题转化成部分的问题，这时候分治、动态规划、暴力搜索这些相应的算法就可以派上用场了。

如果有两个小的数组A1和A2都是beautiful array的话， 那么， 能不能把这两个小的数组合并成一个beautiful array呢？  
如果其中一个都是偶数， 一个都是奇数， 那么合并后一定还是一个beautiful array,
因为本身两个小数组自身都已经是beautiful array了， 所以i,j,k在自己里面找一定不存在，
然后如果是i和j在两个数组里面各取一个的话， 那么结果就是奇数， 而A[k] * 2 是偶数， 所以这一定不存在。  
所以， 只要先构造一个奇数的beautiful array, 再构造一个偶数的beatiful array, 那么左右合并就是一个新的beautiful array。  

因此就可以把序列分成奇数序列和偶数序列，然后合并。

参考的博客：
[http://www.noteanddata.com/leetcode-932-Beautiful-Array-java-solution-note.html](http://www.noteanddata.com/leetcode-932-Beautiful-Array-java-solution-note.html)

这里再贴一下其他写的比较好的讲解：

## **Intuition**

Try to divide and conquer, so we have left part, right part.

One way is to divide into `[1, N / 2]` and `[N / 2 + 1, N]`. But it will cause problems when we merge them.

Another way is to divide into odds part and evens part. So there is no `k` with `A[k] * 2 = odd + even`

I brute force all permutations when N = 5:  
20 beautiful array found, only 4 don't fit odd + even pattern:  
`[2, 1, 4, 5, 3]`  
`[3, 1, 2, 5, 4]`  
`[3, 5, 4, 1, 2]`  
`[4, 5, 2, 1, 3]`  

## **Beautiful Array Properties**

Saying that an array is beautiful, there is no `i < k < j`, such that `A[k] * 2 = A[i] + A[j]`.

Apply these 3 following changes a beautiful array, we can get a new beautiful array.

**1. Deletion**  
Easy to prove. （表明可以将大问题转化为小问题，这个对于计算机实现很重要）

**2. Addition**  
If we have `A[k] * 2 != A[i] + A[j]`,
`(A[k] + x) * 2 = A[k] * 2 + 2x != A[i] + A[j] + 2x = (A[i] + x) + (A[j] + x)`

E.g: `[1,3,2] + 1 = [2,4,3]`.

**3. Multiplication**  
If we have `A[k] * 2 != A[i] + A[j]`,
`(A[k] * x) * 2 = A[k] * 2 * x != (A[i] + A[j]) * x = (A[i] * x) + (A[j] * x)`

E.g: `[1,3,2] * 2 = [2,6,4]`

## **Explanation**

With the observations above, we can easily construct any beautiful array.  
Assume we have a beautiful array `A` with length `N`

`A1 = A * 2 - 1` is beautiful with only odds from `1` to `N * 2 -1`  
`A2 = A * 2` is beautiful with only even from `2` to `N * 2`  
`B = A1 + A2` beautiful array with length `N * 2`  

## **Time Complexity**

I have iteration version here `O(N)` （调和级数，N -> N / 2 -> N /4 -> N / 8...，总的复杂度不超过2N）  
Naive recursion is `O(NlogN)`  
Recursion with one call or with cache is `O(N)`

其他Python解法：

**Sort by reversed binary**：

    def beautifulArray(self, N):
        return sorted(range(1, N + 1), key=lambda x: bin(x)[:1:-1])

**Naive recursion**：

    def beautifulArray(self, N):
        return [i * 2 for i in self.beautifulArray(N / 2)] + [i * 2 - 1 for i in self.beautifulArray((N + 1) / 2)] if N > 1 else [1]

**Binary Reverse**：

    def beautifulArray(self, N):
        return [i for i in [int('{:010b}'.format(i)[::-1], 2) for i in range(1, 1 << 10)] if i <= N]

贴一下自己写的C++代码：

    class Solution {
    public:
        vector<int> beautifulArray(int N) {
            if(N == 1) return {1};
            vector<int> pre = beautifulArray((N & 1) ? (N + 1) / 2 : N / 2);
            vector<int> res;
            for(int i = 0; i < pre.size(); i++) if(2 * pre[i] != N + 1) res.push_back(2 * pre[i]);
            for(int i = 0; i < pre.size(); i++) if(2 * pre[i] - 1 != N + 1) res.push_back(2 * pre[i] - 1);
            return res;
        }
    };

更深入去思考这道题，在分治的时候将序列分成偶数的序列和奇数的序列，同时由于分治的两个部分是相同的，所以可以将子问题个数从2个降低到1个，复杂度也能从O(n log n)降低到O(n)。同时，考虑本题和FFT的关系，实际上相当于是FFT求多项式在N个数值处的值中系数相同的情况，所以可以从O(n log n)降低到O(n)。同时在FFT的递归树的叶子节点从左往右的编号，正好就是题目所需要的一个序列。所以在高效迭代版本的FFT中，进行蝶形操作前预处理的时候，进行了bit-reverse-operation。这也是为什么Python代码直接对1~N的二进制bit反向之后排序就能得到一个可行结果的原因。

同时，beautiful序列的这个不等于的性质里面左侧`2 * a[k]`一定是偶数，而右边奇偶性不确定，这个特点也要注意。a[i]和a[j]一个奇数一个偶数可以使得不定式恒成立。这也是一个突破口。所以就可以把偶数的数字放在一边，奇数的数字放在另一边。这是从上到下的分治法思想。

而前面所说的去看`2 * a[k] != a[i] + a[j]`性质，实际上是`bottom-up`的思考方式，假设我们已经知道了一个序列满足这个性质，能够把它扩展成更大的序列。前面列出的第一条可删除性质就满足这一点。然后后续就需要考察如何将满足这个性质的小序列扩展成更大的序列。不等式满足两边的加法不变性和乘法不变性，分成偶数序列和奇数序列能够满足合并之后这个性质也成立。

能想到将奇数序列和偶数序列的**关键突破点**：不等式左边是偶数，右边奇偶性不确定。此外，从群的角度来看，偶数和奇数序列在加法和乘法下，分别是整个序列的子群，要保证最终结果的封闭性，所以要用if判断一下。

## 934. Shortest Bridge

**题意**：给你一个2D的数组A，它里面有两个岛屿。(一个岛屿是一个全1的区域，可以向四个方向延伸，并且周围都是0)。现在你可以把0变成1，然后让两个岛屿连成1个岛屿。

返回最少需要修改的0的个数，保证答案至少是1。

例如：  
**Input**: [[0,1],[1,0]]  
**Output**: 1  

**Input**: [[0,1,0],[0,0,0],[0,0,1]]  
**Output**: 2  

**Input**: [[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]  
**Output**: 1  

**思路**：既然题目中说了有两个岛屿，就可以用flood-fill先把这两个岛屿都求出来。然后我们考虑要把这两个岛屿连起来。那么岛屿内部的1肯定是不用管的。我们只需要把每个岛屿边界的0改成1即可。所以相当于求两个岛屿的边界点(这些点周围四个点至少有1个点是0)之间的最短距离即可。所以在求距离的时候因为本来就是曼哈顿距离，没必要用bfs去求距离了。

代码如下：

class Solution {
public:
    int shortestBridge(vector<vector<int>>& A) {
        //A = {{1,1,1,1,1},{1,0,0,0,1},{1,0,1,0,1},{1,0,0,0,1},{1,1,1,1,1}};
        int n = A.size(), m = A[0].size();
        vector<vector<int>> cat(n, vector<int>(m, 0)), vis(n, vector<int>(m, 0));
        vector<pair<int,int>> vec1, vec2;
        int idx = 0;
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < m; j++)
            {
                if(!vis[i][j] && A[i][j])
                {
                    idx++;
                    dfs(A, i, j, idx, vis, cat);
                }
            }
        }

        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < m; j++)
            {
                if(A[i][j])
                {
                    int d[][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
                    int flag = 0;
                    for(int k = 0; k < 4; k++)
                    {
                        int ni = i + d[k][0], nj = j + d[k][1];
                        if(ni >= 0 && ni < n && nj >= 0 && nj < m && !A[ni][nj]) {flag = 1; break;}
                    }
                    if(flag)
                    {
                        if(cat[i][j] == 1) vec1.push_back({i, j});
                        else if(cat[i][j] == 2) vec2.push_back({i, j});
                    }
                }
            }
        }
        int ans = 1e9;
        for(int i = 0; i < vec1.size(); i++)
        {
            for(int j = 0; j < vec2.size(); j++)
            {
                int dis = abs(vec1[i].first - vec2[j].first) + abs(vec1[i].second - vec2[j].second) - 1;
                ans = min(ans, dis);
            }
        }
        return ans;
    }

    void dfs(vector<vector<int>>& A, int x, int y, int idx, vector<vector<int>>& vis, vector<vector<int>>& cat)
    {
        int n = A.size(), m = A[0].size();
        int d[][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        vis[x][y] = 1;
        cat[x][y] = idx;
        for(int i = 0; i < 4; i++)
        {
            int nx = x + d[i][0], ny = y + d[i][1];
            if(nx >= 0 && nx < n && ny >= 0 && ny < m && !vis[nx][ny] && A[nx][ny])
            {
                dfs(A, nx, ny, idx, vis, cat);
            }
        }
    }
};

## 935. Knight Dialer

**题意**：给你一个国际象棋的骑士，然后这个棋子可以走的方向是如下左图的几个方向。然后给你一个键盘，如下右图。一开始把棋子放到键盘上的一个有数字的键上，然后让棋子走`N - 1`步，必须保证每一步都得走在带有数字的键上。这样就可以形成N位数字（可以允许前导0），问你总共能够组成多少个不同的N位数字？因为结果可能比较大，所以结果对`10 ^ 9 + 7`取模。

<img src=https://assets.leetcode.com/uploads/2018/10/12/knight.png width=200 height=200> <img src="https://assets.leetcode.com/uploads/2018/10/30/keypad.png" width=200 height=200>

例子：

**Input**: 1  
**Output**: 10  

**Input**: 2  
**Output**: 20  

**Input**: 3  
**Output**: 46  

数据范围：

- `1 <= N <= 5000`

**思路**：首先根据小的数据计算出数字是怎样组成的，可以得到每个数字后面跟的数字集合。然后对于第i位数字可以通过它后面的数字(第i - 1位)的结果都加起来，就得到以这个数字开头的i位数字的总个数。所以递推即可，最终需要求个和。然后计算的时候注意取模。

代码如下：

    class Solution {
    public:
        int knightDialer(int N) {
            if(N == 1) return 10;
            vector<vector<int>> graph = {{4, 6}, {6, 8}, {7, 9}, {4, 8}, {0, 3, 9}, {}, {0, 1, 7}, {2, 6}, {1, 3}, {2, 4}};
            vector<long long> pre(10, 1), nxt(10, 0);
            pre[5] = 0;
            const int mod = 1e9 + 7;
            for(int hop = 1; hop < N; hop++)
            {
                for(int i = 0; i < 10; i++)
                {
                    if(graph[i].size() > 0)
                    {
                        for(int j = 0; j < graph[i].size(); j++)
                        {
                            int num = graph[i][j];
                            nxt[i] = (nxt[i] + pre[num]) % mod;
                        }
                    }
                }
                for(int i = 0; i < 10; i++) pre[i] = nxt[i], nxt[i] = 0; //记着把nxt清零
            }
            long long ans = 0;
            for(int i = 0; i < 10; i++) ans = (ans + pre[i]) % mod;
            return ans;
        }
    };