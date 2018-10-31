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

**题意**：