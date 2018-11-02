# Problem 601~900 Medium

## 647. Palindromic Substrings

**题意**： 给定一个字符串，你的任务是计算有多少个回文子串。对于一个回文子串，只要子串的起始位置和终止位置不同就算做不同的回文子串。

例如：'aaa'的回文子串有6个，3个'a'，2个'aa'，1个'aaa'。

**思路**：这道题可以看做是第5题的变形，这里是统计整个字符串中回文子串的数量。和第5题一样，这道题可以通过dp得到所有回文子串有哪些，然后只需要有一个变量来统计个数即可。

代码如下：

    class Solution {
    public:
        int countSubstrings(string s) {
            int n = s.length(), ans = 0;
            vector<vector<int>> dp(n, vector<int>(n, 0));
            for(int len = 1; len <= n; len++)
            {
                for(int i = 0; i + len - 1 < n; i++)
                {
                    int j = i + len - 1;
                    if(len == 1) dp[i][j] = 1;
                    else if(len == 2 && s[i] == s[j]) dp[i][j] = 1;
                    else if(s[i] == s[j] && dp[i + 1][j - 1]) dp[i][j] = 1;
                    if(dp[i][j]) ans++;
                }
            }
            return ans;
        }
    };

由于都是O(n^2)的复杂度，所以也可以根据回文串的性质采取由中心向两侧扩展的方式来计算总和。分别计算以`i`为中心的奇回文串的个数，和以`(i - 1, i)`为中心的偶回文串的个数。

代码如下：

    int countSubstrings(string s) {
        int res = 0, n = s.length();
        for(int i = 0; i < n; i++){
            for(int j = 0; i-j >= 0 && i+j < n && s[i-j] == s[i+j]; j++)res++; //substring s[i-j, ..., i+j]
            for(int j = 0; i-1-j >= 0 && i+j < n && s[i-1-j] == s[i+j]; j++)res++; //substring s[i-1-j, ..., i+j]
        }
        return res;
    }

## 654. Maximum Binary Tree

**题意**：给一个没有重复数字的整数数组，基于这个数组的最大二叉树的定义为：

- 根节点是数组中的最大元素
- 左子树是由最大元素分割的左半数组形成的最大二叉树
- 右子树是由最大元素分割的右半数组形成的最大二叉树

**思路**：题目给的定义就符合递归的定义。直接dfs即可。这种算法的时间复杂度平均是O(nlog n)，即数组可以分成两半，因此就有着排序的复杂度O(nlog n)。但是也可能退化为一条链，所以复杂度最差为O(n ^ 2)，有点类似于快速排序。  
具体代码如下：  

    class Solution {
    public:
        TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
            int sz = nums.size(), l = 0, r = sz - 1;
            return dfs(nums, l, r);
        }
        TreeNode* dfs(vector<int>& nums, int l, int r)
        {
            if(l > r) return NULL;
            if(l == r) return new TreeNode(nums[l]);
            int pos = -1;
            for(int i = l; i <= r; i++) if(pos == -1 || nums[i] > nums[pos]) pos = i;
            TreeNode* root = new TreeNode(nums[pos]);
            root -> left = dfs(nums, l, pos - 1);
            root -> right = dfs(nums, pos + 1, r);
            return root;
        }
    };

还有一种写法：

    class Solution {
    public:
        TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
            if (nums.size() == 0) return nullptr;
            auto i = nums.cbegin();
            TreeNode* curr = new TreeNode{*i};
            while (++i != nums.cend()) 
            {
                TreeNode* next = new TreeNode{*i};
                if(*i > curr->val) 
                {
                    next->left = curr;
                    curr = next;
                }
                else
                {
                    TreeNode* insert = curr;
                    while (insert->right && insert->right->val > *i) 
                    {
                        insert = insert->right;
                    }
                    next->left = insert->right;
                    insert->right = next;
                }
            }
            return curr;
        }
    };


另外一种做法，使用到了栈。笛卡尔树构造算法。

The key idea is:

1. We scan numbers from left to right, build the tree one node by one step;
2. We use a stack to keep some (not all) tree nodes and ensure a decreasing order;
3. For each number, we keep pop the stack until empty or a bigger number; The bigger number (if exist, it will be still in stack) is current number's root, and the last popped number (if exist) is current number's left child (temporarily, this relationship may change in the future); Then we push current number into the stack.

代码如下：

    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
            vector<TreeNode*> stk;
            for (int i = 0; i < nums.size(); ++i)
            {
                TreeNode* cur = new TreeNode(nums[i]);
                while (!stk.empty() && stk.back()->val < nums[i])
                {
                    cur->left = stk.back();
                    stk.pop_back();
                }
                if (!stk.empty())
                    stk.back()->right = cur;
                stk.push_back(cur);
            }
            return stk.front();
        }
    };

关于该算法是O(N)的证明：

I think a proof can be outlined as follows: Ignoring the nested while-loop, the rest of the work is O(N). So, we just need to show that the total work done by the while-loop across all for-loop iterations add up to O(N). We first note that every node is pushed exactly once and popped at most once. Also noticed that when the while-loop iterates, it always pops a node from the stack. Thus, the total number of iterations by the while-loop is bounded above by the number of pops it can make. Since there are at most N pops, the total work done by the while-loop is O(N).
Thus, the run time is O(N).

然后这道题实际上的数据结构叫做**笛卡尔树**。自己之前是见过的，可以和RMQ进行转换。

## 655. Print Binary Tree

**题意**：打印一棵二叉树。将二叉树存储在$m \* n$的2D string数组中。其中行数m是二叉树的高度，列数n必须是奇数。根节点必须在最中间。

**思路**：可以算出来n应该是$2 ^ h - 1$。相当于是满二叉树的节点个数。然后直接递归打印即可，选好位置。

## 658. Find K Closest Elements

**题意**：给定一个已排序的数组，两个整数k和x，找到数组中k个离x最近的数字。返回的结果也必须按照递增排序。如果存在相等的情形，则更小的数字优先级更高。

其中k是正数，且始终小于等于数组长度。数组长度为正数且不超过`10\^4`。数组元素和x的绝对值不超过`10^4`。

例如：[1, 2, 3, 4, 5], k = 4, x = 3。返回结果为[1, 2, 3, 4]。  
[1, 2, 3, 4, 5], k = 4, x = -1。返回结果为[1, 2, 3, 4]。

**思路**：先二分查找到大于等于x的数字的位置。然后使用双指针查找k个数字。总的复杂度是O(log n + k)。

代码如下：

    class Solution {
    public:
        vector<int> findClosestElements(vector<int>& arr, int k, int x) {
            int sz = arr.size(), l = -1, r = sz;
            vector<int> vec;
            while(r - l > 1)
            {
                int mid = (l + r) / 2;
                if(arr[mid] >= x) r = mid;
                else l = mid;
            }
            int tot = 0;
            while(tot < k)
            {
                while(tot < k && l >= 0 && r < sz)
                {
                    if(x - arr[l] <= arr[r] - x) l--;
                    else r++;
                    tot++;
                }
                while(tot < k && l >= 0) l--, tot++;
                while(tot < k && r < sz) r++, tot++;
            }
            for(int i = l + 1; i < r; i++) vec.push_back(arr[i]);
            return vec;
        }
    };

更简洁的版本：

    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        int index = std::lower_bound(arr.begin(), arr.end(), x) - arr.begin();
        int i = index - 1, j = index;
        while(k--) (i<0 || (j<arr.size() && abs(arr[i] - x) > abs(arr[j] - x) ))? j++: i--;
        return vector<int>(arr.begin() + i + 1, arr.begin() + j );
    }

相比起来，自己双指针的while循环写的就很丑了。。**这里需要学习**。

## 673. Number of Longest Increasing Subsequence

**题意**：给定一个未排序的整数数组，找出最长上升子序列的数量。保证给定数组的长度不超过2000。

例如：[1, 3, 5, 4, 7]，最长上升子序列有[1, 3, 4, 7]和[1, 3, 5, 7]，长度为4，个数为2。

**思路**：朴素的最长上升子序列。然后加上统计个数信息。注意在统计个数的时候，既要保证`dp[j] + 1 == dp[i]`，还要保证元素上升的性质即`nums[j] < nums[i]`才可以。忘了这个就会WA。

因为最长上升子序列有最优子结构的性质，所以不需要去保存每个数字它的前驱数字的二维vector，然后再去搜索。而是利用动态规划的性质在做动态规划的同时就统计好个数。**要加深对于动态规划性质的理解**。自己写的根据每个数字的前驱vector的二维vector做dfs搜索总个数的程序就不贴了。

代码如下：

    class Solution {
    public:
        int findNumberOfLIS(vector<int>& nums) {
            int sz = nums.size();
            vector<int> dp(sz, 0), sum(sz, 0);
            int ans = 0, maxlen = 0;
            for(int i = 0; i < sz; i++)
            {
                dp[i] = 1;
                for(int j = 0; j < i; j++)
                {
                    if(nums[j] < nums[i])
                        dp[i] = max(dp[i], dp[j] + 1);
                }
                for(int j = 0; j < i; j++)
                    if(nums[j] < nums[i] && dp[j] + 1 == dp[i]) sum[i] += sum[j];
                if(!sum[i]) sum[i] = 1;
                maxlen = max(maxlen, dp[i]);
            }
            for(int i = 0; i < sz; i++)
                if(dp[i] == maxlen) ans += sum[i];
            return ans;
        }
    };

另一种循环更少的写法：

    int findNumberOfLIS(vector<int>& nums) {
        int n = nums.size(), res = 0, max_len = 0;
        vector<pair<int,int>> dp(n,{1,1});            //dp[i]: {length, number of LIS which ends with nums[i]}
        for(int i = 0; i<n; i++){
            for(int j = 0; j <i ; j++){
                if(nums[i] > nums[j]){
                    if(dp[i].first == dp[j].first + 1)dp[i].second += dp[j].second;
                    if(dp[i].first < dp[j].first + 1)dp[i] = {dp[j].first + 1, dp[j].second};
                }
            }
            if(max_len == dp[i].first)res += dp[i].second;
            if(max_len < dp[i].first){
                max_len = dp[i].first;
                res = dp[i].second;
            }
        }
        return res;
    }

这个更简洁的版本，动态更新时if判断的顺序很重要。思考不清楚容易出现错误。

当然最长上升子序列也可以用O(nlog n)的版本更快地得到。

这里贴一下别人的解法：

[https://leetcode.com/problems/number-of-longest-increasing-subsequence/discuss/107295/9ms-C++-Explanation:-DP-+-Binary-search-+-prefix-sums-O(NlogN)-time-O(N)-space](https://leetcode.com/problems/number-of-longest-increasing-subsequence/discuss/107295/9ms-C++-Explanation:-DP-+-Binary-search-+-prefix-sums-O(NlogN)-time-O(N)-space)

The idea is to modify classic LIS solution which uses binary search to find the "insertion point" of a currently processed value. At dyn[k] we don't store a single number representing the smallest value such that there exists a LIS of length k+1 as in classic LIS solution. Instead, at dyn[k] we store all such values that were once endings of a k+1 LIS (so we keep the history as well).

These values are held in the first part of the pairs in `vector<pair<int,int>>` which we get by indexing dyn vector. So for example in a pair x = {a, b} the first part -- a, indicates that there exists a LIS of length k+1 such that it ends with a value a. The second part -- b, represents the number of possible options for which LIS of length k+1 ends with a value equal to or greater than a. This is the place where we use prefix sums.

If we want to know how many options do we have to end a LIS of length m with value y, we just binary search for the index i of a pair with first part strictly less than y in dyn[m-2]. Then the number of options is dyn[m-2].back().second - dyn[m-2][i-1].second or just dyn[m-2].back() if i is 0.

That is the basic idea, the running time is O(NlogN), because we just do 2 binary searches for every element of the input. Space complexity is O(N), as every element of the input will be contained in the dyn vector exactly once.

Feel free to post any corrections or simpler explanations :)

    class Solution {
    public:
        int findNumberOfLIS(vector<int>& nums) {
            if (nums.empty())
                return 0;
            vector<vector<pair<int, int>>> dyn(nums.size() + 1);
            int max_so_far = 0;
            for (int i = 0; i < nums.size(); ++i) {
                // bsearch insertion point
                int l = 0, r = max_so_far;
                while (l < r) {
                    int mid = l + (r - l) / 2;
                    if (dyn[mid].back().first < nums[i]) {
                        l = mid + 1;
                    } else {
                        r = mid;
                    }
                }
                // bsearch number of options
                int options = 1;
                int row = l - 1;
                if (row >= 0) {
                    int l1 = 0, r1 = dyn[row].size();
                    while (l1 < r1) {
                        int mid = l1 + (r1 - l1) / 2;
                        if (dyn[row][mid].first < nums[i]) {
                            r1 = mid;
                        } else {
                            l1 = mid + 1;
                        }
                    }
                    options = dyn[row].back().second;
                    options -= (l1 == 0) ? 0 : dyn[row][l1 - 1].second;
                }
                dyn[l].push_back({nums[i], (dyn[l].empty() ? options : dyn[l].back().second + options)});
                if (l == max_so_far) {
                    max_so_far++;
                }
            }
            return dyn[max_so_far-1].back().second;
        }
    };

## 695. Max Area of Island

**题意**：给你一个非空的2D数组，数组元素为0或1。y一块陆地是一些全1的联通分量，联通分量可以向上下左右四个方向延伸。请你找出这个2D数组中最大的连通分量大小。如果数组中没有1，则最大联通分量大小为0。

**思路**：dfs搜索然后再取max即可。

代码如下：

    class Solution {
    public:
        int maxAreaOfIsland(vector<vector<int>>& grid) {
            int n = grid.size(), m = grid[0].size();
            vector<vector<int>> vis(n, vector<int>(m, 0));
            int ans = 0;
            for(int i = 0; i < n; i++)
            {
                for(int j = 0; j < m; j++)
                {
                    if(!vis[i][j] && grid[i][j])
                    {
                        int num = 0;
                        dfs(grid, vis, i, j, num);
                        ans = max(ans, num);
                    }
                }
            }
            return ans;
        }
        void dfs(vector<vector<int>>& grid, vector<vector<int>>& vis, int x, int y, int& num)
        {
            int d[][2] = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
            int n = grid.size(), m = grid[0].size();
            vis[x][y] = 1;
            num++;
            for(int i = 0; i < 4; i++)
            {
                int nx = x + d[i][0], ny = y + d[i][1];
                if(nx >= 0 && nx < n && ny >= 0 && ny < m && grid[nx][ny] && !vis[nx][ny])
                {
                    dfs(grid, vis, nx, ny, num);
                }
            }
        }
    };

## 698. Partition to K Equal Sum Subsets

**题意**：给你一个整数的数组nums，和一个正整数k，判断这个数组能否划分成k个非空子集，并且每个子集的元素之和都相等。

数据范围是：`1 <= k <= len(nums) <= 16`，`0 < nums[i] < 10000`。

**思路**：其实还是蛮难想的，要考虑集合的划分。一开始想用dp每次凑够`sum / k`之后，再删掉这些数字，然后跑k次求sum的dp。但是这样是会出错的。。因为你不知道这一次就得选择这个求和的子集，可能删掉这个子集之后，剩下的子集就凑不够了。这样容易得到错误的结果。

然后考虑具体记录当前有哪些数字被选过了，再看数据范围只有16，所以和二进制表示数据不谋而合，使用二进制来做状态的压缩。正巧这时候选择数字是按照子集选取，而没有顺序选择的问题，正好也是符合状态压缩的。因此这道题就可以用状态压缩DP来做。

我一开始的思路是把所有等于`sum / k`的子集处理出来，然后根据当前的状态去添加这些`sum / k`的子集从而实现状态的转移。同时要保证当前状态不能和`sum / k`中选出来的状态有相同的1，这样就会有冲突。因为最后是要保证每个数字都只选一次。

代码如下：

    class Solution {
    public:
        bool canPartitionKSubsets(vector<int>& nums, int k) {
            int sz = nums.size(), sum = 0;
            for(int i = 0; i < sz; i++) sum += nums[i];
            if(sum % k != 0) return false;
            vector<int> vec;
            for(int i = 0; i < (1 << sz); i++)
            {
                int var = 0;
                for(int j = 0; j < sz; j++)
                {
                    if(i & (1 << j))
                    {
                        var += nums[j];
                    }
                }
                if(var == sum / k) vec.push_back(i);
            }
            int snum = vec.size();
            vector<int> dp(1 << sz, -1);
            dp[0] = 0;
            for(int i = 0; i < snum; i++)
            {
                for(int j = 0; j < (1 << sz); j++)
                {
                    if(!(j & vec[i]) && dp[j] != -1)
                    {
                        dp[j | vec[i]] = dp[j] + 1;
                    }
                }
            }
            return dp[(1 << sz) - 1] == k;
        }
    };

另外的一种直接进行状态压缩DP的方法，这种方法比较巧妙。尤其是能够每次填一个数，还保证都是按照和为`sum / k`去致密地填充。

代码如下：

    class Solution {
    public:
        bool canPartitionKSubsets(vector<int>& nums, int k) {
            //nums = {2,2,2,2,3,4,5}, k = 4;
            int sz = nums.size(), sum = 0;
            for(int i = 0; i < sz; i++) sum += nums[i];
            if(sum % k != 0) return false;
            vector<int> dp(1 << sz, 0), cnt(1 << sz, 0);
            dp[0] = 1;
            int target = sum / k;
            for(int sts = 0; sts < (1 << sz); sts++)
            {
                if(dp[sts])
                {
                    for(int j = 0; j < sz; j++)
                    {
                        int fur = sts | (1 << j), sub = cnt[sts] % target; //这里的这个sub + nums[j] <= target限制非常好。。能够保证每次的target都填完，此外如果这一次填的超过了target，就填不了。所以能保证k次都填满才能到达全1的状态
                    //还有一点是对于该状态fur的来源。。只要fur任意去掉一个1的位置即可，因为它之前的状态都等价。。一个数字nums[j]不管分到哪个组，对该组的贡献都是一样的
                        if(!(sts & (1 << j)) && !dp[fur] && sub + nums[j] <= target)
                        {
                            dp[fur] = 1;
                            cnt[fur] = cnt[sts] + nums[j];
                        }
                    }
                }
            }
            return dp[(1 << sz) - 1];
        }
    };

还有做法通过dfs去搜索全部的方案，复杂度是O(k ^ n)，虽然实际运行速度比动态规划要快。但是这不应该是标准的解法。这里不贴代码了，只记录一下网址。  
[https://leetcode.com/problems/partition-to-k-equal-sum-subsets/discuss/140541/Clear-explanation-easy-to-understand-C++-:-4ms-beat-100](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/discuss/140541/Clear-explanation-easy-to-understand-C++-:-4ms-beat-100)

## 801. Minimum Swaps To Make Sequences Increasing

**题意**：我们有两个整数数组A和B，长度非零。我们可以交换A[i]和B[i]。问你最少交换多少次能够保证A数组和B数组都是严格递增的？（一个序列是严格递增的当且仅当A[0] < A[1] < A[2] < ... < A[A.length - 1]。）返回最少交换的次数，保证数据是存在解的。

A, B数组的长度范围是[1, 1000]，A[i]和B[i]的范围是[0, 2000]。

例如：A = [1, 3, 5, 4]，B = [1, 2, 3, 7]。  
交换A[3]和B[3]，可以得到A = [1, 3, 5, 7], B = [1, 2, 3, 4]。

**思路**：考虑到一开始A, B两个数组不一定是严格递增的，而最终整体的目标是严格递增。而严格递增满足最优子结构，即A, B两个数组整体都严格递增了，则每一个相同的部分都是严格递增的。所以可以把整体的问题转化成若干个子问题来求解。

对于当前第i个位置有两种决策，一种是交换，另一种是不交换。然后就可以将1~i的问题转换成1~(i-1)的子问题和i的子问题。

然后就可以用O(N)的动态规划来计算结果。

代码如下：

    class Solution {
    public:
        int minSwap(vector<int>& A, vector<int>& B) {
            int n = A.size();
            const int inf =  1e9;
            vector<vector<int>> dp(n, vector<int>(2, inf));
            dp[0][0] = 0, dp[0][1] = 1;
            for(int i = 1; i < n; i++)
            {
                if(A[i - 1] < A[i] && B[i - 1] < B[i]) dp[i][0] = min(dp[i][0], dp[i - 1][0]), dp[i][1] = min(dp[i][1], dp[i - 1][1] + 1);
                if(A[i - 1] < B[i] && B[i - 1] < A[i]) dp[i][0] = min(dp[i][0], dp[i - 1][1]), dp[i][1] = min(dp[i][1], dp[i - 1][0] + 1);
            }
            return min(dp[n - 1][0], dp[n - 1][1]);
        }
    };

## 802. Find Eventual Safe States

**题意**：在一个连通图中，我们从每个节点开始，然后每次沿着一条有向边沿着图走到下一个节点。如果我们走到了一个终点（没有从该节点出去的有向边），则我们会停下来。

定义一个节点的状态为'eventually safe'当且仅当我们从这个节点开始最终能走到一个终点。即存在一个自然数K，不管我们从这个节点出发怎么走，都能够在K步之内走到一个终点。

有向图有N个节点0、1、...、N - 1，整个图按照如下格式给出：graph[i]是一个节点j的list，表示(i, j)之间有一条有向边。

graph节点数最多为10000，边数不超过32000。每个graph[i]的list是排好序的数组，数组元素互不相同，且数组元素都在[0, graph.length - 1]的范围内。

例如：  
`Input: graph = [[1,2],[2,3],[5],[0],[5],[],[]]`  
`Output: [2,4,5,6]`

**思路**：这道题说白了就是找出来有哪些节点是在有向图的环上的。不在环上的节点就是最终要输出的答案。

所以自然引出**有向图求环**。有向图求环的方法有**拓扑排序法**和**染色方法**。

染色法的原理是'white-gray-black' DFS Algorithm。即通过设置节点标志来判断节点的染色状态。白色状态表示节点未被访问，灰色节点表示节点在环上，黑色节点表示节点不在环上。

添加反向边然后进行拓扑排序的代码如下：

    class Solution {
    public:
        vector<int> eventualSafeNodes(vector<vector<int>>& graph) {
            int n = graph.size();
            vector<unordered_set<int>> gph(n), rgph(n);
            for(int i = 0; i < n; i++)
            {
                for(int j = 0; j < graph[i].size(); j++)
                {
                    int v = graph[i][j];
                    gph[i].insert(v);
                    rgph[v].insert(i);
                }
            }
            queue<int> que;
            vector<int> safe(n, 0);
            for(int i = 0; i < n; i++)
            {
                if(!gph[i].size()) 
                    que.push(i);
            }
            while(!que.empty())
            {
                int id = que.front();
                que.pop();
                safe[id] = 1;
                for(auto it = rgph[id].begin(); it != rgph[id].end(); it++)
                {
                    int u = *it;
                    gph[u].erase(id);
                    if(!gph[u].size()) que.push(u);
                }
            }
            vector<int> res;
            for(int i = 0; i < n; i++) if(safe[i]) res.push_back(i);
            return res;
        }
    };

染色法代码如下：

    class Solution {
    public:
        vector<int> eventualSafeNodes(vector<vector<int>>& graph) {
            int sz = graph.size();
            vector<int> flag(sz, -1);
            vector<int> res;
            for(int i = 0; i < sz; i++)
                if(!dfs(i, graph, flag))
                    res.push_back(i);
            return res;
        }
        //染色法判断有向图是否有环，同时注意该函数中的两个剪枝的地方，保证复杂度为O(N + E)
        int dfs(int u, vector<vector<int>>& graph, vector<int>& flag)
        {
            if(flag[u] != -1) return flag[u];  //需要加一个剪枝，一旦已经是环的那些节点直接返回
            flag[u] = 1;
            for(auto v : graph[u])
            {
                if(flag[v] == 1 || dfs(v, graph, flag))
                {
                    return flag[u] = 1;  //因为这里是获取u节点的状态，所以一旦找到有环的话，就立即返回
                }
            }
            return flag[u] = 0;
        }
    };

## 807. Max Increase to Keep City Skyline

**题意**：在一个二维数组grid中，每个元素grid\[i]\[j]表示建筑的高度。我们可以给任意数量的建筑增加任意的高度。高度为0也被认为是一个建筑。最后从前后左右四个方向看整个数组的高度要**和原数组保持不变**。

问你所有建筑最高可以增加的高度总和是多少？

**思路**：就找出每行每列的最大值，然后对于非最大值的位置增加高度到它对应的行和列的最大值中最小的那个高度即可，然后求和。

## 809. Expressive Words

**题意**：给你一个字符串S，和一个单词的数组。字符串S中连续相同的字母（个数大于等于3）可以选择小于等于该个数的字母添加若干个得到。问你单词数组中的单词经过这样的扩展操作之后，有几个能够变成字符串S？

**思路**：统计字符串连续字符的个数，然后两两比较即可。假设字符串S中连续字符个数N1，单词中的连续字符个数为N2。在两个字符相等的情形下，`(N1 >= N2 && N1 >= 3) || (N1 == N2 && N1 < 3)`是满足条件的。或者按照自己这样的写法也行`if(num2 < num1 || (num2 > num1 && num2 < 3)) {flag = 0; break;}`。

## 813. Largest Sum of Average

**题意**：把一个数组A切分成最多K个连续的组，则分数是每个组的平均值之和。请问最大的分数是多少？

例如：A = [9, 1, 2, 3, 9]
K = 3。
最好的划分方法是\[9], \[1, 2, 3], \[9]。得到的答案是20。

**思路**：**这题属于比较好的题**。一开始想，最优化的结果是平均值之和，想到K个区间每个区间都包含一个很大的值。但是这样长度不好确定，而且题目说的是最多K个区间...所以排序找到比较大的数字分配到区间，这样肯定是不对的，而且区间长度也不好确定。

由于求A的最多划分成K个子数组的最大平均值之和可以看做一个大问题。然后考虑如何从K - 1个子数组转移到K个子数组，最后一个子数组的长度是需要去枚举的。这样下来整个动态规划的复杂度就是O(K * N ^ 2)。

一定要考虑阶段性，考虑状态是如何一步一步变化的，整个A数组，K个子数组就是很好的状态刻画了。然后考虑它是由什么前驱的状态而得到的，从而找到方法。

代码如下：

    class Solution {
    public:
        double largestSumOfAverages(vector<int>& A, int K) {
            int sz = A.size();
            double dp[sz + 10][sz + 10];
            memset(dp, 0, sizeof(dp));
            vector<int> sum = {A[0]};
            for(int i = 1; i < sz; i++) sum.push_back(sum[i - 1] + A[i]);
            for(int i = 0; i < sz; i++)
            {
                dp[i][0] = 0.0;
                dp[i][1] = 1.0 * sum[i] / (i + 1);
                for(int k = 2; k <= min(i + 1, K); k++)
                {
                    for(int j = k - 2; j < i; j++)
                    {
                        dp[i][k] = max(dp[i][k], dp[j][k - 1] + 1.0 * (sum[i] - sum[j]) / (i - j));
                    }
                }
            }
            return dp[sz - 1][K];
        }
    };

## 814. Binary Tree Pruning

**题意**：给你一个元素值为0或1的二叉树的根节点。把不包含1的子树直接去掉，然后返回。

**思路**：直接O(N)处理即可。不需要对于每个节点在去计算它的子树，这样不仅有大量重复计算，而且复杂度是O(N^2)。

代码如下：

    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        TreeNode* pruneTree(TreeNode* root) {
            dfs(root);
            return root;
        }
        int dfs(TreeNode* &root)
        {
            if(root == NULL) return 0;
            int lnum = dfs(root -> left);
            int rnum = dfs(root -> right);
            if(!lnum) root -> left = NULL;
            if(!rnum) root -> right = NULL;
            return lnum + rnum + (root -> val == 1);
        }
    };

## 816. Ambiguous Coordinates

**题意**：给你一个二维的坐标，像"(1, 3)"或者"(2, 0.5)"这样。然后我们去掉全部的逗号、小数点以及空格，然后得到字符串S。返回一个字符串的列表，表示所有可能的原始坐标。然后对于小数而言，整数部分不能有前导0、小数部分也不能全为0（比如00，0.0，0.00，1.0，001，00.01），而且整数部分不能为空（比如.1）。

返回的结果列表中的字符串可以按照任意顺序，注意返回的字符串的两个坐标之间有一个空格。

**思路**：直接处理字符串，然后枚举两个坐标的分界点，然后再去枚举两个坐标数字小数点的位置。写起来应该会比较繁琐。

官方题解如下，复杂度O(N^3)：

    class Solution { //aw
        public List<String> ambiguousCoordinates(String S) {
            List<String> ans = new ArrayList();
            for (int i = 2; i < S.length()-1; ++i)
                for (String left: make(S, 1, i))
                    for (String right: make(S, i, S.length()-1))
                        ans.add("(" + left + ", " + right + ")");
            return ans;
        }
        public List<String> make(String S, int i, int j) {
            // Make on S.substring(i, j)
            List<String> ans = new ArrayList();
            for (int d = 1; d <= j-i; ++d) {
                String left = S.substring(i, i+d);
                String right = S.substring(i+d, j);
                if ((!left.startsWith("0") || left.equals("0"))
                        && !right.endsWith("0"))
                    ans.add(left + (d < j-i ? "." : "") + right);
            }
            return ans;
        }
    }

## 817. Linked List Components

**题意**：给定一个单链表head，链表中的元素互不相同。给定一个列表G，列表中的值是单链表中值的子集。返回G在链表上的联通块的个数。如果两个连续的数值出现在链表的相邻位置，则这两个值是联通的。

链表长度N的范围是`1 <= N <= 10000`，链表中每个元素值都是在`[0, N - 1]`的范围内。G是链表全部数值的子集，G的长度范围是`1 <= G.length <= 10000`。

例如：  
Input:  
head: 0->1->2->3  
G = [0, 1, 3]  
Output: 2  
Explanation:
0 and 1 are connected, so [0, 1] and [3] are the two connected components.  

Input:  
head: 0->1->2->3->4  
G = [0, 3, 1, 4]  
Output: 2  
Explanation:
0 and 1 are connected, 3 and 4 are connected, so [0, 1] and [3, 4] are the two connected components.

**思路**：直接处理链表即可。用一个hash保存一下G中的值。

代码如下：

    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        int numComponents(ListNode* head, vector<int>& G) {
            unordered_set<int> uset;
            for(int i = 0; i < G.size(); i++) uset.insert(G[i]);
            ListNode *p = head;
            int flag = 0, cnt = 0;
            while(p != NULL)
            {
                int val = p -> val;
                if(uset.count(val))
                {
                    if(!flag) flag = 1, cnt++;
                }
                else flag = 0;
                p = p -> next;
            }
            return cnt;
        }
    };

## 820. Short Encoding of Words

**题意**：给一个单词的数组，可以对这个数组通过参考字符串S和索引的数组A来做编码。例如：单词数组为\['time', 'me', 'bell']，我们可以把它写作`S = 'time#bell#'`和`indexes=[0, 2, 5]`。对于每一个索引我们可以找到对应的单词直到遇到'#'字符为止，表示单词结束。

问你参考字符串S的最短长度是多少？

单词数组的长度范围是`1 <= words.length <= 2000`，单词的长度范围是`1 <= words[i].length <= 7`。每个单词只包含小写字母。

**思路**：注意到一个关键的地方是当一个字符串是另一个字符串的后缀的时候，则可以使得长度减小。所以考虑逆序建立字典树。即按照字符串从末端到开头的顺序建立字典树。这样相同后缀就变成了相同的前缀，这样就能减少长度了。然后统计即可。

自己的代码如下（要注意数组中可能有完全相同的字符串，实际上一开始也可以不给Trie设置int型变量，只要先对数组用hash去重即可）：

    class Solution {
    public:
        int minimumLengthEncoding(vector<string>& words) {
            Trie *rt = new Trie();
            for(auto word: words)
            {
                Trie *p = rt;
                for(int i = word.length() - 1; i >= 0; i--)
                {
                    int id = word[i] - 'a';
                    if(!p -> nxt[id]) p -> nxt[id] = new Trie();
                    p = p -> nxt[id];
                }
                p -> val = 1;
            }
            int ans = 0;
            for(auto word: words)
            {
                Trie *p = rt;
                for(int i = word.length() - 1; i >= 0; i--)
                {
                    int id = word[i] - 'a';
                    p = p -> nxt[id];
                }
                int flag = 0;
                for(int i = 0; i < 26; i++) if(p -> nxt[i]) {flag = 1; break;}
                if(!flag && p -> val) ans += word.length() + 1, p -> val = 0;
            }
            return ans;
        }
        struct Trie
        {
            int val;
            Trie *nxt[26];
            Trie() {val = 0, memset(nxt, 0, sizeof(nxt));}
        };
    };

另外一种做法，直接用hash。因为本题的主要矛盾是去掉那些是别的字符串后缀的字符串。

代码如下：

    class Solution {
    public:
        int minimumLengthEncoding(vector<string>& words) {
            unordered_set<string> uset;
            for(auto word: words) uset.insert(word);
            for(auto word: words)
                for(int i = 1; i < word.length(); i++)
                    uset.erase(word.substr(i, word.length() - i));
            int ans = 0;
            for(auto it = uset.begin(); it != uset.end(); it++)
                ans += (*it).length() + 1;
            return ans;
        }
    };

## 822. Card Flipping Game

**题意**：在一个桌子上有N张卡片，每张卡片的正面和反面都有一个数字（两个数字可能不相同）。在选定一张卡片之后，我们可以翻转任意数量的卡片。如果选定的卡片背面的数字X不在任意一张卡片的正面，那么这个数字X就是good的。问最小的是good的数字是多少？如果不存在，输出0。

数据范围如下：

1. `1 <= fronts.length == backs.length <= 1000`.
2. `1 <= fronts[i] <= 2000`.
3. `1 <= backs[i] <= 2000`.

**思路**：我们可以发现只有当有正面和反面两个数字一样的时候，其他数字在翻到底面的时候，才不能够good。如果一个数字，没有正反面一样的话，因为能翻若干个卡片，所以总能够把正面朝上的相同数字翻下去。所以只用考虑**当一个卡片正反面数字一样的时候**，它会使得这个数字不是good的。

代码如下：

    class Solution {
    public:
        int flipgame(vector<int>& fronts, vector<int>& backs) {
            int n = fronts.size();
            unordered_map<int, int> umap;
            for(int i = 0; i < n; i++)
            {
                if(!umap.count(fronts[i])) umap[fronts[i]] = 1;
                if(!umap.count(backs[i])) umap[backs[i]] = 1;
                if(fronts[i] == backs[i]) umap[fronts[i]] = 0;
            }
            int ans = 0;
            for(auto it = umap.begin(); it != umap.end(); it++)
            {
                int num = it -> first, flag = it -> second;
                if(flag) ans = !ans ? num : min(ans, num);
            }
            return ans;
        }
    };

## 823. Binary Trees With Factors

**题意**：给你一个有不同数字的数组，每个整数都严格大于1。我们使用这些数字来构建二叉树，每个数字可以使用任意多次。每个非叶子节点的值必须等于它的孩子的乘积。我们能构建多少个这样的二叉树？返回的答案需要对`10^9 + 7`取模。

例如：  
Input: A = [2, 4, 5, 10]  
Output: 7  
Explanation: We can make these trees: [2], [4], [5], [10], [4, 2, 2], [10, 2, 5], [10, 5, 2].  

数据范围：

1. `1 <= A.length <= 1000`.
2. `2 <= A[i] <= 10 ^ 9`.

**思路**：首先要注意数字可以重复使用。所以对于一个数字而言，需要找到当前数组中有哪些数字之积等于该数字。为了降低`O(N^3)`的复杂度，所以对于每个数字而言，将已访问过的数字使用hash表缓存，然后查询商是否在hash表中即可。将时间复杂度降低到`O(N^2)`。然后注意数字比较大要取模，最好还是将最终的结果定义成long long。然后取模的式子老老实实写，不要去`+=`。

从本题中也可以发现，大的数字作为二叉树的根节点的话，它要是由更小的数字构成的话，方法是就是自己因子对应的个数的乘积。这实际上就把大的问题转化成了子问题。。然后使用dp的方式求出结果。

这里面预先排序也是很重要的，使得能够先算出小问题的答案，然后再得到大问题的答案。

DP的时间复杂度为O(N^2)。

代码如下：

    class Solution {
    public:
        int numFactoredBinaryTrees(vector<int>& A) {
            int n = A.size();
            sort(A.begin(), A.end());
            unordered_map<int, int> umap;
            vector<int> dp(n, 1);
            long long ans = 0;
            const int mod = 1e9 + 7;
            for(int i = 0; i < n; i++)
            {
                umap[A[i]] = i;
                for(int j = 0; j < i; j++)
                {
                    if(A[i] % A[j] == 0 && umap.count(A[i] / A[j]))
                    {
                        int id = umap[A[i] / A[j]];
                        dp[i] = (dp[i] + 1LL * dp[j] * dp[id]) % mod;
                    }
                }
                ans = (ans + dp[i]) % mod;
            }
            return ans;
        }
    };

## 825. Friends Of Appropriate Ages

**题意**：一些人可以发起朋友请求，age数组表示每个人的年龄。

A不向B(B != A)发起朋友请求当且仅当以下的任意一个条件成立：

- `age[B] <= 0.5 * age[A] + 7`
- `age[B] > age[A]`
- `age[B] > 100 && age[A] < 100`

否则，A将会向B发起朋友请求。
如果A向B发起了朋友请求，但是B不是必须要向A发起朋友请求。同时，每个人也不会向自己发起朋友请求。

问最终总共有多少个朋友请求？

数据范围：

- `1 <= ages.length <= 20000`
- `1 <= ages[i] <= 120`

**思路**：题目意思很绕。。orz。。还容易出错。

首先是求出三个条件的反面，即A向B发起请求的条件。  
`age[B] > 0.5 * age[A] + 7 && age[B] <= age[A] && (age[B] <= 100 || age[A] > 100)`  
这里需要注意`age[B] > 0.5 * age[A] + 7`并且`age[B] <= age[A]`，则可以得到`age[A] > 0.5 * age[A] + 7`，即`age[A] > 14`。这是一个隐含条件。

所以总的条件可以变成`ages[A] > 14 && age[B] > 0.5 * age[A] + 7 && age[B] <= age[A]`。

自己最新的代码(36ms)，优化了if...else的判断：

    class Solution {
    public:
        int numFriendRequests(vector<int>& ages) {
            int n = ages.size(), ans = 0;
            unordered_map<int, int> umap;
            for(int i = 0; i < n; i++) umap[ages[i]]++;
            for(auto i = umap.begin(); i != umap.end(); i++)
            {
                int a = i -> first, anum = i -> second;
                for(auto j = umap.begin(); j != umap.end(); j++)
                {
                    int b = j -> first, bnum = j -> second;
                    if(a == b && anum == 1) continue;
                    if(a > 14 && b > 0.5 * a + 7 && b <= a) ans += anum * (bnum - (a == b)); //这里是优化的地方，如果写成if...else...的形式，则时间为48ms
                }
            }
            return ans;
        }
    };

自己之前写的比较蠢的代码(68ms)，没有看清楚age数组元素的取值范围那么小，用二分查找屈才了：

    class Solution {
    public:
        int numFriendRequests(vector<int>& ages) {
            sort(ages.begin(), ages.end());
            int sz = ages.size();
            vector<int> preN(sz), cntN(130);
            for(int i = 0; i < sz; i++)
            {
                if(i == 0) preN[i] = -1;
                else if(ages[i - 1] < ages[i]) preN[i] = i - 1;
                else preN[i] = preN[i - 1];
                cntN[ages[i]]++;
            }
            int ans = 0;
            for(int i = 0; i < sz; i++)
            {
                int pos = upper_bound(ages.begin(), ages.begin() + preN[i] + 1, floor(0.5 * ages[i] + 7)) - ages.begin();
                if(ages[i] <= 14) continue;
                if(pos < preN[i] + 1) ans += (preN[i] + 1 - pos);
                if(cntN[ages[i]] > 1) ans += (cntN[ages[i]] - 1);
            }
            return ans;
        }
    };

别人家的代码：

    int numFriendRequests(vector<int>& ages) {
        int a[121] = {}, res = 0;
        for (auto age : ages) ++a[age];
        for (auto i = 15; i <= 120; ++i)
            for (int j = 0.5 * i + 8; j <= i; ++j) res += a[j] * (a[i] - (i == j));
        return res;
    }

自己写的哈希表实际上和这里的数组的作用是一样的。

别人家优化的版本：

    int numFriendRequests(vector<int>& ages) {
        int a[121] = {}, res = 0;
        for (auto age : ages) ++a[age];
        for (auto i = 15, minAge = 15, sSum = 0; i <= 120; sSum += a[i], res += a[i++] * (sSum - 1))
            while (minAge <= 0.5 * i + 7) sSum -= a[minAge++];
        return res;
    }

这个sliding sum(类似移动窗口)的优化做法，只需要每次剔除掉那些不在下界之上的数字，然后上界不用管，因为如果去掉while循环的话本来就是求1 ~ i - 1的sum和。

## 835. Image Overlap

**题意**：给定两个图片A和B，用2D的01矩阵表示，两个矩阵是大小相同的方阵。我们平移(平移的方向是上下左右)一张图片，然后把它放在另一张图片上方。然后，这次平移的overlap就是两张图片中公共的1的个数。

问最大可能的overlap是多少？

**思路**：自己的思路就是朴素的暴力枚举A的顶点与B的哪个点重合。然后就有四种情况。O(N^4)计算即可。

代码如下(自己写的太繁琐)：

    class Solution {
    public:
        int largestOverlap(vector<vector<int>>& A, vector<vector<int>>& B) {
            int n = A.size(), m = A[0].size();
            int res = 0;
            for(int i = 0; i < n; i++)
            {
                for(int j = 0; j < m; j++)
                {
                    res = max(res, check(A, B, i, j));
                }
            }
            return res;
        }
        int check(vector<vector<int>>& A, vector<vector<int>>& B, int px, int py)
        {
            int res = 0, sum = 0;
            int n = A.size(), m = A[0].size();
            for(int i = 0; i <= px; i++)
            {
                for(int j = 0; j <= py; j++)
                {
                    int ax = n - 1 - px + i, ay = m - 1 - py + j;
                    if(A[ax][ay] == 1 && B[i][j] == 1) sum++;
                }
            }
            res = max(res, sum);
            sum = 0;
            for(int i = px; i <= n - 1; i++)
            {
                for(int j = 0; j <= py; j++)
                {
                    int ax = i - px, ay = m - 1 - py + j;
                    if(A[ax][ay] == 1 && B[i][j] == 1) sum++;
                }
            }
            res = max(res, sum);
            sum = 0;
            for(int i = 0; i <= px; i++)
            {
                for(int j = py; j <= m - 1; j++)
                {
                    int ax = n - 1 - px + i, ay = j - py;
                    if(A[ax][ay] == 1 && B[i][j] == 1) sum++;
                }
            }
            res = max(res, sum);
            sum = 0;
            for(int i = px; i <= n - 1; i++)
            {
                for(int j = py; j <= m - 1; j++)
                {
                    int ax = i - px, ay = j - py;
                    if(A[ax][ay] == 1 && B[i][j] == 1) sum++;
                }
            }
            res = max(res, sum);
            return res;
        }
    };

官方的题解有两种解法：

- 首先考虑平移的偏移向量delta，对于每个可能的偏移，去计算相应的Overrlap，这样的时间复杂度是O(n^6)。

Java代码如下：

    import java.awt.Point;

    class Solution {
        public int largestOverlap(int[][] A, int[][] B) {
            int N = A.length;
            List<Point> A2 = new ArrayList(), B2 = new ArrayList();
            for (int i = 0; i < N*N; ++i) {
                if (A[i/N][i%N] == 1) A2.add(new Point(i/N, i%N));
                if (B[i/N][i%N] == 1) B2.add(new Point(i/N, i%N));
            }

            Set<Point> Bset = new HashSet(B2);
            int ans = 0;
            Set<Point> seen = new HashSet();
            for (Point a: A2) for (Point b: B2) {
                Point delta = new Point(b.x - a.x, b.y - a.y);
                if (!seen.contains(delta)) {
                    seen.add(delta);
                    int cand = 0;
                    for (Point p: A2)
                        if (Bset.contains(new Point(p.x + delta.x, p.y + delta.y)))
                            cand++;
                    ans = Math.max(ans, cand);
                }
            }
            return ans;
        }
    }

- 第二种做法是第一种做法的逆向思维，因为最终求的是重叠部分相同的1的个数。所以我们只考虑A的1与B的1重叠的情形。相应的就能算出来偏移量。然后相当于对每个偏移量去计数就可以了。时间复杂度O(N^4)，**代码很精简，值得学习**！

Java代码如下：

    class Solution {
        public int largestOverlap(int[][] A, int[][] B) {
            int N = A.length;
            int[][] count = new int[2*N+1][2*N+1];
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    if (A[i][j] == 1)
                        for (int i2 = 0; i2 < N; ++i2)
                            for (int j2 = 0; j2 < N; ++j2)
                                if (B[i2][j2] == 1)
                                    count[i-i2 +N][j-j2 +N] += 1;
            int ans = 0;
            for (int[] row: count)
                for (int v: row)
                    ans = Math.max(ans, v);
            return ans;
        }
    }

还有一些基于第二种做法的更精简的代码，这里只放网址：

[https://leetcode.com/problems/image-overlap/discuss/130623/C++JavaPython-Straight-Forward](https://leetcode.com/problems/image-overlap/discuss/130623/C++JavaPython-Straight-Forward)  

## 841. Keys and Rooms

**题意**：有N个屋子，你一开始在第0个屋子。每个屋子都有一个不同的数字0,1,2,..., N-1。每个屋子有一些到下一个屋子的钥匙。每个屋子i都有一个钥匙的列表`rooms[i]`，每个钥匙`rooms[i][j]`是一个范围在`[0, 1, ..., N-1]`的整数。其中`N = rooms.length`。一个钥匙`rooms[i][j] = v`可以打开第v个屋子。

一开始的时候，除了0号屋子以外，其他屋子都是锁上的。你可以来回在屋子之间走。如果你可以走到每一个屋子，返回`true`。否则返回`false`。

**思路**：无向图求联通分量，直接dfs即可。

代码如下：

    class Solution {
    public:
        bool canVisitAllRooms(vector<vector<int>>& rooms) {
            int sz = rooms.size();
            vector<int> flag(sz);
            dfs(0, rooms, flag);
            for(int i = 0; i < sz; i++)
            {
                if(!flag[i])
                    return false;
            }
            return true;
        }
        void dfs(int u, vector<vector<int>>& rooms, vector<int>& flag)
        {
            flag[u] = 1;
            for(auto v: rooms[u])
            {
                if(!flag[v])
                {
                    dfs(v, rooms, flag);
                }
            }
        }
    };

## 845. Longest Mountain in Array

**题意**：我们把一个子数组称为`mountain`当且仅当满足以下的性质：

- `B.length >= 3`
- 存在某个`0 < i < B.length - 1`满足`B[0] < B[1] < ... < B[i - 1] < B[i] > B[i + 1] > ... > B[B.length - 1]`

B可以是A的任意子数组，包括完整的A数组。  
给你一个全是整数的A数组，返回最长的`mountain`的长度。  
如果没有`mountain`，则返回0。

数据范围：

- `0 <= A.length <= 10000`
- `0 <= A[i] <= 10000`

要求：

- 能否只使用一次循环？
- 能否只用O(1)的空间？

**思路**：就直接普通的判断就好了。实际上有贪心的性质。

    class Solution {
    public:
        int longestMountain(vector<int>& A) {
            int sz = A.size();
            int ans = 0, i = 0;
            while(i < sz)
            {
                int p = i;
                while(p + 1 < sz && A[p + 1] > A[p]) p++;
                if(p > i)
                {
                    int q = p;
                    while(p + 1 < sz && A[p + 1] < A[p]) p++;
                    if(p > q)
                    {
                        ans = max(ans, p - i + 1);
                        i = p;
                    }
                    else
                        i++;
                }
                else
                {
                    i++;
                }
            }
            return ans;
        }
    };

## 853. Car Fleet

**题意**：N个汽车都要驶向同一个终点，终点离原点的距离为`target`英里。每个汽车i都有一个恒定的速度`speed[i]`(英里/小时)，同时给出每个汽车的初始位置`position[i]`。

**每个汽车永远都不能超过它前面的车**，但是它能追上前面的车，同时车速与前面的车速相同，可以认为这两辆车在同一个位置上。

一个`car fleet`指的是在同一时刻速度相同的汽车的非空集合，一辆汽车也可以构成一个`car fleet`。如果一辆车在终点的时候才追上另一辆车，那么也可以认为它们构成一个`car fleet`。

问你到达终点的时候总共有多少个`car fleet`？

数据范围：

1. `0 <= N <= 10 ^ 4`
2. `0 < target <= 10 ^ 6`
3. `0 < speed[i] <= 10 ^ 6`
4. `0 <= position[i] < target`
5. 所有初始位置都互不相同

**思路**：实际上需要先忽略其他汽车对自己造成的影响，看当前的汽车匀速行驶多长时间能够到达重点。对于每辆车都能计算出行驶时间。然后考虑路程时间图像，去考虑每辆汽车怎样才会形成车队，当pos更小的汽车，所用的时间比pos大的汽车时间短的时候，就会形成车队。根据这个特点来统计个数即可。

自己的代码如下，写的很丑，其实没必要用DSU：

    class Solution {
    public:
        int carFleet(int target, vector<int>& position, vector<int>& speed) {
            vector<pair<int,int>> vec;
            int sz = position.size();
            if(sz == 0) return 0;
            for(int i = 0; i < sz; i++) vec.push_back(make_pair(position[i], speed[i]));
            sort(vec.begin(), vec.end());
            vector<pair<double, int>> time(sz);
            const double eps = 1e-6;
            vector<int> par(sz, 0);
            for(int i = 0; i < sz; i++) time[i] = make_pair(1.0 * (target - vec[i].first) / vec[i].second, i),  par[i] = i;
            sort(time.begin(), time.end());
            int i = sz - 1;
            while(i >= 0)
            {
                int p = i - 1, idi = time[i].second;
                while(p >= 0 && time[p].second < idi)
                {
                    int idp = time[p].second;
                    int x = find(idp, par), y = find(idi, par);
                    par[x] = y;
                    p--;
                }
                i = p;
            }
            int res = 0;
            for(int i = 0; i < sz; i++) if(par[i] == i) res++;
            return res;
        }
        int find(int x, vector<int>& par)
        {
            if(par[x] == x) return x;
            return par[x] = find(par[x], par);
        }
    };

官方题解，实际上没必要用栈。**关键点是排序，然后贪心**。

Java代码如下：

    class Solution {
        public int carFleet(int target, int[] position, int[] speed) {
            int N = position.length;
            Car[] cars = new Car[N];
            for (int i = 0; i < N; ++i)
                cars[i] = new Car(position[i], (double) (target - position[i]) / speed[i]);
            Arrays.sort(cars, (a, b) -> Integer.compare(a.position, b.position));
            int ans = 0, t = N;
            while (--t > 0) {
                if (cars[t].time < cars[t-1].time) ans++; //if cars[t] arrives sooner, it can't be caught
                else cars[t-1] = cars[t]; //else, cars[t-1] arrives at same time as cars[t]
            }
            return ans + (t == 0 ? 1 : 0); //lone car is fleet (if it exists)
        }
    }
    class Car {
        int position;
        double time;
        Car(int p, double t) {
            position = p;
            time = t;
        }
    }

别人家的五行代码，思路超级清晰，值得学习：

    int carFleet(int target, vector<int>& pos, vector<int>& speed, int fl = 0, double time = 0) {
        map<int, double> m;
        for (auto i = 0; i < pos.size(); ++i) m[pos[i]] = ((double)target - pos[i]) / speed[i];
        for (auto it = m.rbegin(); it != m.rend(); ++it) {
            if (it->second > time) ++fl, time = it->second;
        }
        return fl;
    }

## 856. Score of Parentheses

**题意**：给你一个匹配的括号序列S，计算整个序列的分数。规则如下：

- ()有1分
- AB的分数是A+B，其中A和B是匹配的括号序列
- (A)的分数是2 \* A，其中A是匹配的括号序列

例如：

    Input: "(()(()))"  
    Output: 6  

数据范围：

1. S是匹配的括号序列，且只包含'('和')'
2. 2 <= S.length <= 50

**思路**：自己的思路是先用栈找出匹配的位置，然后再去dfs计算。这样能保证算法复杂度是O(N)的，而不是O(N^2)。

代码如下：

    class Solution {
    public:
        int scoreOfParentheses(string S) {
            int sz = S.length();
            stack<int> st;
            vector<int> pos(sz, -1);
            for(int i = 0; i < sz; i++)
            {
                if(S[i] == '(') st.push(i);
                else
                {
                    pos[st.top()] = i;
                    st.pop();
                }
            }
            return dfs(S, 0, sz - 1, pos);
        }
        int dfs(string& S, int l, int r, vector<int>& pos)
        {
            int i = l, ans = 0;
            while(i <= r)
            {
                if(pos[i] - 1 - i >= 2)
                    ans += 2 * dfs(S, i + 1, pos[i] - 1, pos);
                else
                    ans++;
                i = pos[i] + 1;
            }
            return ans;
        }
    };

官方题解是用栈可以直接搞：

自己没看官方栈版本写的：

    class Solution {
    public:
        int scoreOfParentheses(string S) {
            int sz = S.length();
            vector<int> st;
            int ans = 0;
            for(int i = 0; i < sz; i++)
            {
                if(S[i] == '(') st.push_back(0);
                else
                {
                    int num = st.back();
                    st.pop_back();
                    if(!num)
                    {
                        if(!st.size()) ans++;
                        else st[st.size() - 1]++;
                    }
                    else
                    {
                        if(!st.size()) ans += 2 * num;
                        else st[st.size() - 1] += 2 * num;
                    }
                }
            }
            return ans;
        }
    };

官方的Java版本代码：

    public int scoreOfParentheses(String S) {
        Stack<Integer> stack = new Stack();
        stack.push(0); // The score of the current frame
        for (char c: S.toCharArray()) {
            if (c == '(')
                stack.push(0);
            else {
                int v = stack.pop();
                int w = stack.pop();
                stack.push(w + Math.max(2 * v, 1));
            }
        }
        return stack.pop();
    }

确实写的要比自己的版本简洁。使用max来判断到底是取2\*v还是取1。

官方的第三种解法，直接统计。代码如下：

    class Solution {
        public int scoreOfParentheses(String S) {
            int ans = 0, bal = 0;
            for (int i = 0; i < S.length(); ++i) {
                if (S.charAt(i) == '(') {
                    bal++;
                } else {
                    bal--;
                    if (S.charAt(i-1) == '(')
                        ans += 1 << bal;
                }
            }
            return ans;
        }
    }

这两种官方的解法都是考虑了左括号的深度。因此可以直接加和计算。

## 858. Mirror Reflection

**题意**：有一个正方形的屋子，四面墙上都有一面镜子。除了左下角之外，每个顶点都有一个受体。右下角受体的编号是0，右上角受体的编号是1，左上角受体的编号是2。

屋子的长度是p，有一束光从左下角发射打到右侧墙上的位置和右下角顶点间的距离是q。让你返回灯光最先打到的受体编号是多少？(数据保证解存在)

数据范围：

1. `1 <= p <= 1000`
2. `0 <= q <= p`

[https://leetcode.com/problems/mirror-reflection/description/](https://leetcode.com/problems/mirror-reflection/description/)

**思路**：高中物理镜面反射题。。顺便用一用gcd。自己当时想着去模拟整个反射的过程，太麻烦了。。实际上需要以右侧墙壁的镜面将屋子对称反射到右侧。

官方题解如下：

    class Solution {
        public int mirrorReflection(int p, int q) {
            int g = gcd(p, q);
            p /= g; p %= 2;
            q /= g; q %= 2;
            if (p == 1 && q == 1) return 1;
            return p == 1 ? 0 : 2;
        }

        public int gcd(int a, int b) {
            if (a == 0) return b;
            return gcd(b % a, a);
        }
    }

因为数据保证了解的存在性，所以不用考虑p = 0, q是0还是非0。因为只要p=0的时候，肯定第一次碰到的是2所在的左侧墙壁的顶点。

## 861. Score After Flipping Matrix

**题意**：有一个二维的01矩阵A。每一步操作可以选择任意行或列，然后翻转那一行或列的每一个数字，把所有的0变成1，把所有的1变成0。

进行了一系列的操作之后，把每一行看做一个二进制数字，总的分数就是所有行的二进制数字之和。

返回最大的数字。

**思路**：为了保证数字最大，肯定贪心地先把最高位都变成1，只需要`A[i][0] == 0`的行做翻转即可。然后考虑每一列，这时候第一列肯定全是1了。考虑剩下的列是否需要变换，当这一列中0的个数大于1的个数，则对这一列进行翻转之后，0变成了1，所以翻转之后1的个数就会比0的个数要多。

代码如下：

    class Solution {
    public:
        int matrixScore(vector<vector<int>>& A) {
            int n = A.size(), m = A[0].size();
            for(int i = 0; i < n; i++)
            {
                if(!A[i][0])
                    for(int j = 0; j < m; j++) A[i][j] ^= 1;
            }
            for(int j = 1; j < m; j++)
            {
                int cnt = 0;
                for(int i = 0; i < n; i++) cnt += A[i][j];
                if(cnt < n - cnt)
                    for(int i = 0; i < n; i++) A[i][j] ^= 1;
            }
            int ans = 0;
            for(int i = 0; i < n; i++)
            {
                int num = 0;
                for(int j = 0; j < m; j++)
                    num += A[i][j] * (1 << (m - 1 - j));
                ans += num;
            }
            return ans;
        }
    };

## 865. Smallest Subtree with all the Deepest Nodes

**题意**：给定一个二叉树的根节点root，各个节点的深度是该节点到根节点的最短距离。

一个节点是深度最深的当且仅当它的深度是整个树的节点中最深的。

返回所有深度最深的节点集合深度最深的公共祖先。

数据范围：

- 树中节点的个数范围是1~500
- 每个节点的值是唯一的

**思路**：一开始求最大深度是很好想的。接下来如何判断所有深度最深的节点的位置是比较困难的。。之前的做法是计算出每个节点的子树中最深节点的个数，进而通过判断左子树的个数和右节点的个数与总个数的关系来判断如何继续向下遍历。

这样实际上是没有必要的，我们得到了最大深度之后，对于根节点而言，如果左右子树的最大深度都等于整棵树的最大深度，则直接返回根节点。否则如果只有左子树的最大深度为整棵树的最大深度的话，则继续dfs搜索左子树。反之，搜索右子树。

有点脑筋急转弯的感觉，这次做差点没转过来弯。算法复杂度是O(N^2)。

代码如下：

    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        TreeNode* subtreeWithAllDeepest(TreeNode* root) {
            int maxh = getHeight(root);
            return dfs(root, maxh);
        }
        int getHeight(TreeNode* root)
        {
            if(root == NULL) return 0;
            return max(getHeight(root -> left), getHeight(root -> right)) + 1;
        }
        TreeNode* dfs(TreeNode* root, int maxh)
        {
            if(root == NULL) return NULL;
            int lh = getHeight(root -> left) + 1, rh = getHeight(root -> right) + 1;
            if(lh == maxh && rh == maxh) return root;
            if(lh == maxh) return dfs(root -> left, maxh - 1);
            if(rh == maxh) return dfs(root -> right, maxh - 1);
        }
    };

官方题解提供了两种O(N)的做法，第一种做法是通过hash对我的这种方法每个节点的高度进行了存储。第二种后序遍历的方法值得学习。第二种方法的原理就是动态获取高度并且维护最深的节点位置。

代码如下（自己用C++改写）：

    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Result
    {
    public:
        TreeNode* node;
        int h;
        Result(TreeNode* _nd, int _h)
        {
            node = _nd, h = _h;
        }
    };

    class Solution {
    public:
        TreeNode* subtreeWithAllDeepest(TreeNode* root) {
            return dfs(root).node;
        }

        Result dfs(TreeNode* root)
        {
            if(root == NULL) return Result(NULL, 0);
            Result L = dfs(root -> left), R = dfs(root -> right);
            if(L.h > R.h) return Result(L.node, L.h + 1);
            if(R.h > L.h) return Result(R.node, R.h + 1);
            return Result(root, L.h + 1);
        }
    };

## 869. Reordered Power of 2

**题意**：给你一个正整数N，对数字的每一位进行重排(重排后不能有前导0)。判断这个数字重排之后能否等于2的幂？

数据范围：`1 <= N <= 10^9`

**思路**：对于这种重排之后比较大小的为题，一般有两种方法，第一种是统计每个字符(每一位的数字)的个数。然后比较个数的向量是否完全相同。另一种对于字符串重排而言，可以对字符串sort排序，然后使得字符串的字符都是按照字典序递增排列，然后比较排序后的字符串是否相等。

官方题解代码：

    class Solution {
        public boolean reorderedPowerOf2(int N) {
            int[] A = count(N);
            for (int i = 0; i < 31; ++i)
                if (Arrays.equals(A, count(1 << i)))
                    return true;
            return false;
        }

        // Returns the count of digits of N
        // Eg. N = 112223334, returns [0,2,3,3,1,0,0,0,0,0]
        public int[] count(int N) {
            int[] ans = new int[10];
            while (N > 0) {
                ans[N % 10]++;
                N /= 10;
            }
            return ans;
        }
    }

时间复杂度是$O(log^{2} N)$，总共有`logN`个不同的2的幂，每次比较需要$O(log N)$的时间复杂度。

## 873. Length of Longest Fibonacci Subsequence

**题意**：一个序列`X_1, X_2, ..., X_n`是`fibonacci-like`的当且仅当

- `n >= 3`
- 对于所有`i + 2 <= n`都有`X_i + X_{i + 1} = X_{i + 2}`

给定一个全是正数组成的**严格递增**的数组A，找到A中最长的`fibonacci-like`子序列的长度。如果不存在，则返回0。

数据范围：

- `3 <= A.length <= 1000`
- `1 <= A[0] < A[1] < A[A.length - 1] <= 10^9`

例如：  

    Input: [1,3,7,11,12,14,18]
    Output: 3
    Explanation:
    The longest subsequence that is fibonacci-like:
    [1,11,12], [3,11,14] or [7,11,18].

**思路**：这道题实际上是在斐波那契数列的基础上进行的一个扩展。对于斐波那契数列，我们只要知道两个数字，就能知道整个数列的信息。所以对于阶段性的划分而言只要当前数字和前一个数字，两个数字相减就能得到更前面的数字。这样我们只需要两个状态即可。然后就可以转化为序列dp。

这道题实际上也是序列dp的变种，最原始的序列dp有最长上升子序列。因而对于斐波那契的子序列而言整体的问题可以变为少一个数字的子问题，然后长度+1即可。

代码如下：

    class Solution {
    public:
        int lenLongestFibSubseq(vector<int>& A) {
            int n = A.size();
            vector<vector<int>> dp(n, vector<int>(n, 0));
            unordered_map<int, int> hash;
            int ans = 0;
            for(int i = 0; i < n; i++)
            {
                for(int j = i - 1; j >= 0; j--)
                {
                    int pre = A[i] - A[j];
                    if(hash.count(pre) && hash[pre] < j) dp[i][j] = max(dp[i][j], dp[j][hash[pre]] + 1);
                    else dp[i][j] = 2;
                    ans = max(ans, dp[i][j]);
                }
                hash[A[i]] = i;
            }
            if(ans < 3) ans = 0;
            return ans;
        }
    };

但是上面的代码运行速度比较慢，100多ms。改了一些部分之后，可以跑到64ms。

代码如下：

    class Solution {
    public:
        int lenLongestFibSubseq(vector<int>& A) {
            int n = A.size();
            int dp[n + 10][n + 10];
            memset(dp, 0, sizeof(dp));
            unordered_map<int, int> hash;
            for(int i = 0; i < n; i++) hash[A[i]] = i;
            int ans = 0;
            for(int i = 0; i < n; i++)
            {
                dp[i][i] = 1;
                for(int j = 0; j < i; j++)
                {
                    if(A[i] < 2 * A[j] && hash.count(A[i] - A[j]))
                    {
                        int idx = hash[A[i] - A[j]];
                        dp[i][j] = max(dp[i][j], dp[j][idx] + 1);
                    }
                    else dp[i][j] = 2;
                    ans = max(ans, dp[i][j]);
                }
            }
            if(ans < 3) ans = 0;
            return ans;
        }
    };

这个代码里面if语句先判断A[i]与A[j]的关系可以防止每次都去查询hash表，从而节省时间。因为数字不会重复，所以可以用数值来判断位置大小。然后if语句里面单独把hash的value用idx保存也能加快时间。

而且预先把数据存在hash表中，dp的时候直接查询会比dp的时候动态查询哈希表，动态插入更快。。很奇怪。。

> 最奇怪的是加了`dp[i][i] = 1`这行语句之后，程序运行速度反而更快了。。能提高20ms的速度。

## 877. Stone Game

**题意**：Alex和Lee玩N堆石子的游戏，N堆石子摆成一行，总的堆数是偶数，每堆石子`piles[i]`都是正数。

谁最终拿到的石子最多谁获胜。N堆石子的总数是奇数，保证不会出现平局的情况。

Alex和Lee轮流拿石子，Alex先拿。每次拿的时候只能从最左边或者最右边拿完整的一堆石子。直到最后没有剩余的石子为止，石子最多的人获胜。

假设Alex和Lee都是足够聪明的，如果Alex获胜，返回True。反之，返回False。

数据范围：

1. `2 <= piles.length <= 500`
2. `piles.length` is even.
3. `1 <= piles[i] <= 500`
4. `sum(piles)` is odd.

**思路**：简单博弈题。这种题的思路一般是先从小的数据开始考虑，然后进行推广。主要矛盾在于奇数堆的总和与偶数堆的总和。先手有必胜的策略。

当然也可以使用dp来求解，这个是题解提供的解法。

**Intuition**：

Let's change the game so that whenever Lee scores points, it deducts from Alex's score instead.

Let `dp(i, j)` be the largest score Alex can achieve where the piles remaining are `piles[i], piles[i+1], ..., piles[j]`. This is natural in games with scoring: we want to know what the value of each position of the game is.

We can formulate a recursion for `dp(i, j)` in terms of `dp(i+1, j)` and `dp(i, j-1)`, and we can use dynamic programming to not repeat work in this recursion. (This approach can output the correct answer, because the states form a DAG (directed acyclic graph).)

**Algorithm**：

When the piles remaining are `piles[i], piles[i+1], ..., piles[j]`, the player who's turn it is has at most 2 moves.

The person who's turn it is can be found by comparing `j-i` to `N` modulo 2.

If the player is Alex, then she either takes `piles[i]` or `piles[j]`, increasing her score by that amount. Afterwards, the total score is either `piles[i] + dp(i+1, j)`, or `piles[j] + dp(i, j-1)`; and we want the maximum possible score.

If the player is Lee, then he either takes `piles[i]` or `piles[j]`, decreasing Alex's score by that amount. Afterwards, the total score is either `-piles[i] + dp(i+1, j)`, or `-piles[j] + dp(i, j-1)`; and we want the *minimum* possible score.

    class Solution {
    public:
        bool stoneGame(vector<int>& piles) {
            int N = piles.size();

            // dp[i+1][j+1] = the value of the game [piles[i], ..., piles[j]]
            int dp[N+2][N+2];
            memset(dp, 0, sizeof(dp));

            for (int size = 1; size <= N; ++size)
                for (int i = 0, j = size - 1; j < N; ++i, ++j) {
                    int parity = (j + i + N) % 2;  // j - i - N; but +x = -x (mod 2)
                    if (parity == 1)
                        dp[i+1][j+1] = max(piles[i] + dp[i+2][j+1], piles[j] + dp[i+1][j]);
                    else
                        dp[i+1][j+1] = min(-piles[i] + dp[i+2][j+1], -piles[j] + dp[i+1][j]);
                }

            return dp[1][N] > 0;
        }
    };

这实际上用dp实现的min-max算法的过程。一个人走DAG min的分支，一个人走max的分支。

## 881. Boats to Save People

**题意**:第i个人的体重是`people[i]`，每艘船可以承载的最大重量为`limit`。每艘船最多可以同时载两个人，这两个人的体重不能超过最大载重。

求最小的船只数量能够载下每个人。(保证每个人都能被一艘船载下，即答案是存在的)

数据范围：

1. `1 <= people.length <= 50000`
2. `1 <= people[i] <= limit <= 30000`

**思路**：肯定是先要排序的。然后先放重量大的人，因为最多放两个人，所以剩下的空隙去放重量最小的人。自然想到双指针。

代码如下：

    class Solution {
    public:
        int numRescueBoats(vector<int>& people, int limit) {
            sort(people.begin(), people.end());
            int num = 0, l = 0, r = people.size() - 1;
            while(l <= r)
            {
                int load = 0;
                if(load + people[r] <= limit) load += people[r], r--;
                if(load + people[l] <= limit) load += people[l], l++;
                num++;
            }
            return num;
        }
    };

## 885. Spiral Matrix III

**题意**：