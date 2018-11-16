# Problem 901~1200 Medium

## 901. Online Stock Span

**题意**：写一个`StockSpanner`类，能够收集每只股票日常的价格，然后返回当天股票价格的*span*。

今天股票价格的*span*定义为：从当天之前的某一天开始价格小于或等于当天价格的连续的天数。

比如：如果一支股票一周7天的股票是`[100, 80, 60, 70, 60, 75, 85]`，所以股票的*span*是`[1, 1, 1, 2, 1, 4, 6]`。

例子：

**Input**: ["StockSpanner","next","next","next","next","next","next","next"], [[],[100],[80],[60],[70],[60],[75],[85]]  
**Output**: [null,1,1,1,2,1,4,6]  
**Explanation**:  
First, S = StockSpanner() is initialized.  Then:  
S.next(100) is called and returns 1,  
S.next(80) is called and returns 1,  
S.next(60) is called and returns 1,  
S.next(70) is called and returns 2,  
S.next(60) is called and returns 1,  
S.next(75) is called and returns 4,  
S.next(85) is called and returns 6.  

Note that (for example) S.next(75) returned 4, because the last 4 prices  
(including today's price of 75) were less than or equal to today's price.  

数据范围：

1. `StockSpanner.next(int price)`的调用中满足`1 <= price <= 10^5`  
2. `StockSpanner.next`每组测试数据上最多调用`10000`次
3. `StockSpanner.next`在所有测试数据上最多调用`150000`次

**思路**：朴素地去思考，对于每一个位置，都要往前去找长度。这样的时间复杂度是O(n^2)。但是这样肯定会超时的，为了加速可以采用跳跃的方法设置left数组，然后在O(n)的时间内求出答案。

代码如下：

    class StockSpanner {
    public:
        StockSpanner() {
            prices.clear();
            left.clear();
        }
        int next(int price) {
            int sz = prices.size();
            prices.push_back(price);
            left.push_back(sz - 1);
            int idx = sz;
            while(left[idx] >= 0 && prices[left[idx]] <= prices[sz]) idx = left[idx];
            left[sz] = left[idx];
            return sz - left[sz];
        }
        vector<int> prices, left;
    };

    /**
    * Your StockSpanner object will be instantiated and called as such:
    * StockSpanner obj = new StockSpanner();
    * int param_1 = obj.next(price);
    */

当然，也可以使用单调栈来实现这个功能，实际上本质含义是一样的，只不过是用left数组记录单调栈每个元素对应的边界的位置而已。

## 904. Fruit Into Baskets

**题意**：有一排树，第i棵树产生的水果种类是tree\[i]。

你可以从任意一棵树开始，然后重复执行以下的操作。

1. 将这棵树上的一个水果加入到你的篮子里，如果不能加入篮子，则停止操作。
2. 移动到当前这棵树右边的树，如果右边的树不存在，则停止操作。

在执行操作时，必须按照1,2,1,2...的顺序执行。

你有两个篮子，每个篮子可以装无限的水果，但是每个篮子只能装一种水果。

问你用这两个篮子最多可以装多少个水果？

例子：  
Input: [1,2,3,2,2]  
Output: 4  
Explanation: We can collect [2,3,2,2].  
If we started at the first tree, we would only collect [1, 2].  

Input: [3,3,3,1,2,1,1,2,3,3,4]  
Output: 5  
Explanation: We can collect [1,2,1,1,2].  
If we started at the first tree or the eighth tree, we would only collect 4 fruits.  

数据范围：

1. `1 <= tree.length <= 40000`
2. `0 <= tree[i] < tree.length`

**思路**：先朴素地去思考，对于每一个位置开始往后搜索长度，在搜索的这段长度里只有两种类型的数字。然后再考虑能否在此基础上优化。



## 907. Sum of Subarray Minimums

**题意**：给你一个整数的数组A，找到min(B)的sum和，其中B是A的每个连续的子数组。

因为答案可能会很大，所以答案要对10^9+7取模。

例子：

Input: [3,1,2,4]  
Output: 17  
Explanation: Subarrays are [3], [1], [2], [4], [3,1], [1,2], [2,4], [3,1,2], [1,2,4], [3,1,2,4].  
Minimums are 3, 1, 2, 4, 1, 1, 2, 1, 1, 1.  Sum is 17.  

数据范围：  

1. `1 <= A.length <= 30000`
2. `1 <= A[i] <= 30000`

**思路**：因为数据范围是比较大的，所以不能肯定不能直接去暴力求出每一个子数组。所以只能暴力枚举每一个数字为min(B)的值，然后去求B数组的个数，进而求解。然后就能想到求left和right数组，但是要注意重复元素的影响。在求left数组时计算`>=`的左边界，right数组计算`<`的右边界，然后才能保证计数的答案不会重复。如果right数组也是计算`<=`的右边界的话，就会有重复计算。

比如`71 55 82 55 63`这样重复的情况。因为55右边大于等于55的会计算右边的55,55左边也大于等于55，则会计算左边的那个。。就会有重复区间。。

因此要保证**偏序性**。同时要注意此题与891题(Hard)的区别，同时要注意防止重复的情形。

代码如下：

    class Solution {
    public:
        int sumSubarrayMins(vector<int>& A) {
            int sz = A.size();
            vector<int> left(sz, 0), right(sz, 0);
            left[0] = 0, right[sz - 1] = sz - 1;
            for(int i = 1; i < sz; i++)
            {
                int j = i;
                while(j && A[j - 1] >= A[i]) j = left[j - 1];
                left[i] = j;
            }
            for(int i = sz - 2; i >= 0; i--)
            {
                int j = i;
                while(j < sz - 1 && A[j + 1] > A[i]) j = right[j + 1];
                right[i] = j;
            }
            long long ans = 0;
            const int mod = 1e9 + 7;
            for(int i = 0; i < sz; i++)
            {
                ans = (ans + 1LL * A[i] * (i - left[i] + 1) * (right[i] - i + 1)) % mod;
            }
            return ans;
        }
    };

另一种写法，维护一个单调栈（来自官方题解）。

**Intuition**

For a specific `j`, let's try to count the minimum of each subarray `[i, j]`. The intuition is that as we increment `j++`, these minimums may be related to each other. Indeed, `min(A[i:j+1]) = min(A[i:j], A[j])`.

Playing with some array like `A = [1,7,5,2,4,3,9]`, with `j = 6` the minimum of each subarray `[i, j]` is `B = [1,2,2,2,3,3,9]`. We can see that there are critical points `i = 0, i = 3, i = 5, i = 6` where a minimum is reached for the first time when walking left from `j`.

**Algorithm**

Let's try to maintain an RLE (run length encoding) of these critical points `B`. More specifically, for the above `(A, j)`, we will maintain `stack = [(val=1, count=1), (val=2, count=3), (val=3, count=2), (val=9, count=1)]`, that represents a run length encoding of the subarray minimums `B = [1,2,2,2,3,3,9]`. For each `j`, we want `sum(B)`.

As we increment `j`, we will have to update this stack to include the newest element `(val=x, count=1)`. We need to pop off all `values >= x` before, as the minimum of the associated subarray `[i, j]` will now be `A[j] instead of what it was before.

At the end, the answer is the dot product of this stack: \sum\limits_{\text{e} \in \text{ stack}} \text{e.val} * \text{e.count}, which we also maintain on the side as the variable `dot`.

和一种方法一样，使用单调栈确定每个数字为最小值能够往前延伸的长度。只考虑往前延伸这样也能够防止重复。

自己写的代码如下：

    class node
    {
    public:
        int val, num;
        node() {}
        node(int _v, int _n) {val = _v, num = _n;}
    };

    class Solution {
    public:
        int sumSubarrayMins(vector<int>& A) {
            int n = A.size();
            const int mod = 1e9 + 7;
            long long ans = 0, sum = 0;
            stack<node> st;
            for(int i = 0; i < n; i++)
            {
                int cnt = 1;
                while(!st.empty() && st.top().val >= A[i])
                {
                    sum -= 1LL * st.top().val * st.top().num;
                    cnt += st.top().num;  //这里实际上要加上top元素的num，而不是cnt++
                    st.pop();
                }
                st.push(node(A[i], cnt));
                sum += 1LL * A[i] * cnt;
                ans += sum;
                ans %= mod;
            }
            return ans;
        }
    };

## 909. Snakes and Ladders

**题意**：给你一个N \* N的木板，数字1~N\*N是从模板底部从左往右写，然后下一行再交换写的方向。一个6\*6的木板如下：

![](https://assets.leetcode.com/uploads/2018/09/23/snakes.png)

你在开始的时候位于`1`的位置(最后一行第一列)，每一步从`x`开始，包括以下的步骤：

- 你可以选择一个目标的方块`S`写着数字`x + 1`，`x + 2`，`x + 3`，`x + 4`，`x + 5`或者`x + 6`，假设这个数字`<= N * N`。
- 如果`S`有一条蛇或者梯子，你可以直接移动到那条蛇或者梯子的终点。否则你只能移动到`S`。

一个正方形木板在第`r`行和第`c`列上有一条蛇或者一个梯子的条件是`board[r][c] != -1`，那条蛇或者那个梯子的目标是`board[r][c]`。

注意到你只能搭一条蛇或者一个梯子一次，如果这一次到达的终点是另一条蛇或者另一个梯子的起点，你**不能**再移动到下一个终点。

返回到达方块`N * N`所需要的最小移动步数。如果不能移动到，返回`-1`。

例子：

**Input**: [  
[-1,-1,-1,-1,-1,-1],  
[-1,-1,-1,-1,-1,-1],  
[-1,-1,-1,-1,-1,-1],  
[-1,35,-1,-1,13,-1],  
[-1,-1,-1,-1,-1,-1],  
[-1,15,-1,-1,-1,-1]]  
**Output**: 4  
**Explanation**:  
At the beginning, you start at square 1 [at row 5, column 0].  
You decide to move to square 2, and must take the ladder to square 15.  
You then decide to move to square 17 (row 3, column 5), and must take the snake to square 13.  
You then decide to move to square 14, and must take the ladder to square 35.  
You then decide to move to square 36, ending the game.  
It can be shown that you need at least 4 moves to reach the N*N-th square, so the answer is 4.  

数据范围：

1. `2 <= board.length = board[0].length <= 20`
2. `board[i][j]`的取值在`1`到`N * N`的范围内，或者等于`-1`
3. `1`所在的方块没有蛇或者梯子
4. `N*N`所在的方块没有蛇或者梯子

**思路**：BFS搜索题，容易写错。

自己的代码如下：

    class Solution {
    public:
        int snakesAndLadders(vector<vector<int>>& board) {
            int n = board.size();
            vector<int> vis(n * n + 10, 0);
            vector<pair<int, int>> pos(n * n + 10);
            vector<vector<int>> mat(n, vector<int>(n, 0));
            int tot = 0;
            for(int i = 0; i < n; i++)
            {
                int l = 0, r = n - 1;
                if(i & 1) l = n - 1, r = 0;
                for(int j = l; (i & 1) ? j >= r : j <= r; (i & 1) ? j-- : j++)
                    pos[++tot] = make_pair(n - 1 - i, j);
            }
            queue<pair<int, int>> que;
            que.push(make_pair(1, 0));
            vis[1] = 1;
            int res = -1;
            while(!que.empty())
            {
                auto top = que.front();
                que.pop();
                int id = top.first, step = top.second;
                // cout << id << endl;
                if(id == n * n)
                {
                    res = (res == -1) ? step : min(res, step);
                }
                auto pr = pos[id];
                int px = pr.first, py = pr.second;
                if(board[px][py] != -1) id = board[px][py];
                //cout << id << endl;
                if(id == n * n)
                {
                    res = (res == -1) ? step : min(res, step);
                }
                //cout << "-------" << endl;
                for(int i = 1; i <= 6; i++)
                {
                    int x = id + i;
                    if(x <= n * n && !vis[x])
                    {
                        vis[x] = 1;
                        que.push(make_pair(x, step + 1));
                    }
                }
            }
            return res;
        }
    };

## 910. Smallest Range II

**题意**：给你一个整数的数组`A`，对于每个整数`A[i]`，我们需要选择`x = -K`或者`x = K`，然后将`x`加到`A[i]`上(x只加一次)。

这个操作结束后，得到序列`B`。  

返回`B`数组最大值和最小值的最小可能的差值。

例子：

Input: A = [1,3,6], K = 3  
Output: 3  
Explanation: B = [4,6,3]  

数据范围：

1. `1 <= A.length <= 10000`
2. `0 <= A[i] <= 10000`
3. `0 <= K <= 10000`

**思路**：一开始肯定还是想到先从小到大排序。

好难啊。。个人感觉像是hard难度。。实际上这个还是得秉承固定最小值，去计算最大值的思想。
所以对于每一个i而言，A[i] + k和A[i] - k都有可能成为最小值。然后去判断最大值可能在哪些地方出现。。
还要注意相等的情形，以及边界情形。。

自己的写法(错了很多次)：

    class Solution {
    public:
        int smallestRangeII(vector<int>& A, int k) {
            int sz = A.size();
            if(sz == 1) return 0;
            sort(A.begin(), A.end());
            int l = 0, r = sz - 1;
            int ans = A[r] - A[l];

            for(int i = 0; i < sz; i++)
            {
                if(i > 0 && A[l] >= A[i] - 2 * k) // A[i] - k is min
                {
                    if(i < r) ans = min(ans, max(A[i - 1] + k, A[r] - k) - A[i] + k);
                    else ans = min(ans, A[i - 1] - A[i] + 2 * k);
                }
                if(A[l] == A[i])  // A[i] + k is min
                {
                    ans = min(ans, A[r] - A[i]); //一开始初始化的时候有
                    if(i < r)
                    {
                        int pos = lower_bound(A.begin() + i + 1, A.end(), A[i] + 2 * k) - A.begin();
                        if(pos < sz)
                        {
                            ans = min(ans, max(A[pos - 1] + k, A[r] - k) - A[i] - k);
                        }
                    }
                }
            }
            return ans;
        }
    };

注意官方题解的解法：
#### Intuition

As in Smallest Range I, smaller `A[i]` will choose to increase their value ("go up"), and bigger `A[i]` will decrease their value ("go down").

#### Algorithm

We can formalize the above concept: if `A[i] < A[j]`, we don't need to consider when` A[i]` goes down while `A[j]` goes up. This is because the interval `(A[i] + K, A[j] - K)` is a subset of `(A[i] - K, A[j] + K)` (here, `(a, b)` for `a > b` denotes `(b, a)` instead.)

That means that it is never worse to choose `(up, down)` instead of `(down, up)`. We can prove this claim that one interval is a subset of another, by showing both `A[i] + K` and `A[j] - K` are between `A[i] - K` and `A[j] + K`.

For sorted `A`, say `A[i]` is the largest `i` that goes up. Then `A[0] + K, A[i] + K, A[i+1] - K, A[A.length - 1] - K` are the only relevant values for calculating the answer: every other value is between one of these extremal values.

    class Solution {
        public int smallestRangeII(int[] A, int K) {
            int N = A.length;
            Arrays.sort(A);
            int ans = A[N-1] - A[0];

            for (int i = 0; i < A.length - 1; ++i) {
                int a = A[i], b = A[i+1];
                int high = Math.max(A[N-1] - K, a + K);
                int low = Math.min(A[0] + K, b - K);
                ans = Math.min(ans, high - low);
            }
            return ans;
        }
    }

这个方法实际上通过分析得出整个序列不会出现分段增、分段减、再分段增等奇怪的情况。用反证法简单分析思考一下即可。然后就知道一定是**前一段递增，后一段递减**。这时只要**枚举递增和递减的分界点**即可。然后就是两段有序的序列，就可以直接得到最大值最小值。

## 911. Online Election

**题意**：在一次选举中，第`i`个投票在`times[i]`时间被投给`persons[i]`这个人。

现在我们想要实现以下的查询函数：`TopVotedCandidate.q(int t)`将返回在`t`时刻最领先的人的总个数。

在当前时间`t`的投票也被算在当前的询问中。在平局的情况下(即票数相等)，最近投票的人获胜。

例子：

Input: ["TopVotedCandidate","q","q","q","q","q","q"], [[[0,1,1,0,0,1,0],[0,5,10,15,20,25,30]],[3],[12],[25],[15],[24],[8]]  
Output: [null,0,1,1,0,0,1]  
Explanation:  
At time 3, the votes are [0], and 0 is leading.  
At time 12, the votes are [0,1,1], and 1 is leading.  
At time 25, the votes are [0,1,1,0,0,1], and 1 is leading (as ties go to the most recent vote.)  
This continues for 3 more queries at time 15, 24, and 8.  

数据范围：

1. `1 <= persons.length = times.length <= 5000`  
2. `0 <= persons[i] <= persons.length`  
3. `times`是一个严格递增的序列，所有元素在`[0, 10^9]`范围内
4. 每个测试数据`TopVotedCandidate.q`被调用最多`10000`次
5. `TopVotedCandidate.q(int t)`总是在`t >= times[0]`的情况下调用

**思路**：肯定是动态统计每个人的票数情况，然后用哈希表即可。然后对于每个时间要给出最大票数的人。这就涉及到用可以维护最大值的数据结构，同时可以直接修改最大值。所以用map更方便，**不要因为一开始的维护最大值就想着用堆**。对于当前的票数，map直接修改该票数为当前的人，更好满足了在票数相等的条件下，使用最近票数的人的题意。

其次要**注意在做query的时候，访问的时间t不是递增的**，而一开始给的**time序列是递增的**。所以要用二分查找去time序列中找位置。

这是一道好题。

自己写的代码如下：

    class TopVotedCandidate {
    public:
        vector<int> res, tvec;
        TopVotedCandidate(vector<int> pvec, vector<int> tm) {
            int sz = tm.size();
            res = vector<int>(sz, 0);
            tvec = vector<int>(tm.begin(), tm.end());
            unordered_map<int, int> umap;
            map<int, int> Mp;
            umap[pvec[0]]++;
            Mp[1] = pvec[0];
            res[0] = pvec[0];
            for(int i = 1; i < sz; i++)
            {
                umap[pvec[i]]++;
                Mp[umap[pvec[i]]] = pvec[i];
                auto it = --Mp.end();
                res[i] = it -> second;
            }
        }

        int q(int t) {
            int sz = tvec.size();
            int pos = lower_bound(tvec.begin(), tvec.end(), t) - tvec.begin();
            if(pos == sz) pos--;
            else if(tvec[pos] > t) pos--;
            return res[pos];
        }
    };

/**
 * Your TopVotedCandidate object will be instantiated and called as such:
 * TopVotedCandidate obj = new TopVotedCandidate(persons, times);
 * int param_1 = obj.q(t);
 */

但实际上最大票数的人可以直接用两个变量维护，不用map，这样可以进一步保证预处理的时间是O(n)。这个是参考了题解的第二种做法。

题解的Java代码如下：

    class TopVotedCandidate {
        List<Vote> A;
        public TopVotedCandidate(int[] persons, int[] times) {
            A = new ArrayList();
            Map<Integer, Integer> count = new HashMap();
            int leader = -1;  // current leader
            int m = 0;  // current number of votes for leader

            for (int i = 0; i < persons.length; ++i) {
                int p = persons[i], t = times[i];
                int c = count.getOrDefault(p, 0) + 1;
                count.put(p, c);

                if (c >= m) {
                    if (p != leader) {  // lead change
                        leader = p;
                        A.add(new Vote(leader, t));
                    }
                    if (c > m) m = c;
                }
            }
        }

        public int q(int t) {
            int lo = 1, hi = A.size();
            while (lo < hi) {
                int mi = lo + (hi - lo) / 2;
                if (A.get(mi).time <= t)
                    lo = mi + 1;
                else
                    hi = mi;
            }
            return A.get(lo - 1).person;
        }
    }

    class Vote {
        int person, time;
        Vote(int p, int t) {
            person = p;
            time = t;
        }
    }

## 915. Partition Array into Disjoint Intervals

**题意**：给你一个数组`A`，将它划分成两个连续的子数组`left`和`right`满足：  

- `left`中的每个元素都小于等于`right`中的每个元素
- `left`和`right`是非空的
- `left`的大小尽可能保证最小

返回这样划分之后`left`的长度。保证这样的划分存在。

例子：

Input: [5,0,3,8,6]  
Output: 3  
Explanation: left = [5,0,3], right = [8,6]  

Input: [1,1,1,0,6,12]  
Output: 4  
Explanation: left = [1,1,1,0], right = [6,12]  
数据范围：

1. `2 <= A.length <= 30000`
2. `0 <= A[i] <= 10^6`
3. 保证`A`中至少有一种上述的划分方式存在

**思路**：因为考虑到数组的划分，所以每一个划分的分界点很重要。而考虑到了遍历每一个分界点。接下来左数组的所有元素都小于等于右数组的所有元素，这个等价于**左数组的最大值小于等于右数组的最小值**。

代码如下：

    class Solution {
    public:
        int partitionDisjoint(vector<int>& A) {
            int sz = A.size();
            vector<int> lmax(sz, 0), rmin(sz, 1e8);
            lmax[0] = A[0];
            for(int i = 1; i < sz; i++)
                lmax[i] = max(lmax[i - 1], A[i]);
            rmin[sz - 1] = A[sz - 1];
            for(int i = sz - 2; i >= 0; i--)
                rmin[i] = min(rmin[i + 1], A[i]);
            int pos = 0;
            for(int i = sz - 2; i >= 0; i--)
            {
                if(lmax[i] <= rmin[i + 1])
                {
                    pos = i;
                }
            }
            return pos + 1;
        }
    };

## 916. Word Subsets

**题意**：给你两个单词的数组`A`和`B`，每个单词是小写字母的字符串。现在我们说单词`b`是单词`a`的子集当`b`中每个字母都在`a`中出现，包括出现多次。比如：`"wrr"`是`"warrior"`的一个子集，但是不是`"world"`的子集。

现在说数组`A`中的一个单词`a`是*universal*的如果对于数组`B`中每个单词`b`都是`a`的一个子集。

返回`A`中所有的*universal*的单词，你可以按照任何顺序返回单词。

Input: A = ["amazon","apple","facebook","google","leetcode"], B = ["e","o"]  
Output: ["facebook","google","leetcode"] 

Input: A = ["amazon","apple","facebook","google","leetcode"], B = ["lo","eo"]  
Output: ["google","leetcode"]  

Input: A = ["amazon","apple","facebook","google","leetcode"], B = ["ec","oc","ceo"]  
Output: ["facebook","leetcode"]  

数据范围：

1. `1 <= A.length, B.length <= 10000`
2. `1 <= A[i].length, B[i].length <= 10`
3. `A[i]`和`B[i]`只包含小写字母
4. `A`中的所有单词都是互不相同的，即不存在`i != j`满足`A[i] == A[j]`

**思路**：直接处理即可，先把B中所有的单词存在哈希表中，记录每个字符出现的**最大次数**(这里要注意！！实际上是对B数组的每个单词的状态**求并集**)。然后去A中判断每个单词的哈希表，保证每个字符的次数大于等于对应的B数组得到的哈希表该字符的次数。

代码如下：

    class Solution {
    public:
        vector<string> wordSubsets(vector<string>& A, vector<string>& B) {
            vector<string> res;
            vector<int> cnt(26, 0);
            for(int i = 0; i < B.size(); i++)
            {
                vector<int> bcnt(26, 0);
                for(int j = 0; j < B[i].size(); j++)
                {
                    bcnt[B[i][j] - 'a']++;
                }
                for(int j = 0; j < 26; j++)
                    cnt[j] = max(cnt[j], bcnt[j]);
            }

            for(int i = 0; i < A.size(); i++)
            {
                vector<int> acnt(26, 0);
                for(int j = 0; j < A[i].size(); j++)
                {
                    acnt[A[i][j] - 'a']++;
                }
                int flag = 1;
                for(int j = 0; j < 26; j++)
                {
                    if(cnt[j] > acnt[j]) {flag = 0; break;}
                }
                if(flag) res.push_back(A[i]);
            }
            return res;
        }
    };

## 918. Maximum Sum Circular Subarray

**题意**：给你一个由整数组成的**循环数组C**，用`A`数组表示。找到循环数组**C**中最大可能的非空子数组的sum和。

这里的循环数组表示数组的末尾和数组的开头连接在一起。(即当`0 <= i < A.length`时`C[i] = A[i]`，当`i >= 0`时`C[i + A.length] = C[i]`。)

一个子数组表示只包含A数组中的元素最多一次。(即对于一个子数组`C[i], C[i + 1], ..., C[j]`，不存在`i <= k1, k2 <= j` 使得`k1 % A.length = k2 % A.length`。)

例子：

Input: [3,-1,2,-1]  
Output: 4  
Explanation: Subarray [2,-1,3] has maximum sum 2 + (-1) + 3 = 4

Input: [3,-2,2,-3]  
Output: 3  
Explanation: Subarray [3] and [3,-2,2] both have maximum sum 3  

数据范围：

1. `-30000 <= A[i] <= 30000`
2. `1 <= A.length <= 30000`

**思路**：一开始的思路肯定是对整个数组复制一份追加到末尾，然后再取前缀和。然后部分和就可以转换成在满足`i - j + 1 <= n`的条件下`sum[i] - sum[j]`的最大值。然后就可以使用单调队列来求解。

代码如下：

    class Solution {
    public:
        int maxSubarraySumCircular(vector<int>& A) {
            int n = A.size();
            vector<int> vec(2 * n + 1, 0), que(2 * n + 1, 0);
            for(int i = 1; i <= 2 * n; i++)
                vec[i] = vec[i - 1] + A[(i - 1) % n];
            int st = -1, ed = -1, res = A[0];
            for(int i = 0; i <= 2 * n - 1; i++)
            {
                while(st < ed && que[st + 1] < i - n) st++;
                if(st < ed)
                    res = max(res, vec[i] - vec[que[st + 1]]);
                while(st < ed && vec[que[ed]] >= vec[i]) ed--;
                que[++ed] = i;
            }
            return res;
        }
    };

注意从多个角度来思考本题，注意发散思维。本题实际上是求解循环数组中的最大子数组和，但是子数组的长度小于等于N。

## 919. Complete Binary Tree Inserter

**题意**：给你一颗完全二叉树，让你往里面动态插入新的元素，每次插入一个元素的时候，返回这个元素的父节点的value。

**思路**：这道题还是很不错的。因为涉及到插入新元素，而且插入顺序也都是从左向右。为了保证高效插入，不能每次插入的时候使用O(N)的复杂度去查找插入的位置。考虑到满二叉树的性质，只和最后的两行有关，然后把最后两层的叶子节点暴力找出来，分别存到这两行对应的两个queue里。然后前一行的queue里的节点会作为父节点，然后新插入的节点会成为该父节点的孩子节点。当父节点的两个孩子都插入时，从queue的头部删掉该front节点。当前一行的父节点都被用完时，即前一行的queue的size为0时，说明这时的树是满二叉树了。所以把当前行作为前一行，相当于交换两个queue。为了保证高效，queue用指针，只用交换指针即实现了交换两个queue。

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

    class CBTInserter {
    public:
        queue<TreeNode*> *pre, *nxt;
        TreeNode *nd;
        CBTInserter(TreeNode* root) {
            nd = root;
            pre = new queue<TreeNode *>;
            nxt = new queue<TreeNode *>;
            int dept = getHeight(root);
            dfs(root, 1, dept);
            if(pre -> size() == 0) swap(pre, nxt);
        }

        int getHeight(TreeNode* root)
        {
            if(root == NULL) return 0;
            return max(getHeight(root -> left), getHeight(root -> right)) + 1;
        }

        void dfs(TreeNode *root, int dept, int height)
        {
            if(root == NULL) return;
            if(dept == height - 1 && (root -> left == NULL || root -> right == NULL)) pre -> push(root);
            if(dept == height) nxt -> push(root);
            dfs(root -> left, dept + 1, height);
            dfs(root -> right, dept + 1, height);
        }

        int insert(int v) {
            TreeNode *node = pre -> front(), *newnode = new TreeNode(v);
            nxt -> push(newnode);
            if(node -> left == NULL) node -> left = newnode;
            else
            {
                node -> right = newnode;
                pre -> pop();
                if(pre -> size() == 0) swap(pre, nxt);
            }
            return node -> val;
        }

        TreeNode* get_root() {
            return nd;
        }
    };

    /**
    * Your CBTInserter object will be instantiated and called as such:
    * CBTInserter obj = new CBTInserter(root);
    * int param_1 = obj.insert(v);
    * TreeNode* param_2 = obj.get_root();
    */

题解中的做法是：



## 921. Minimum Add to Make Parentheses Valid

**题意**：给你一个不合法的括号序列，问你最少加多少个括号之后能让这个序列变得合法。

**思路**：老题型了，用栈处理即可。对于右括号不合法的情形，直接结果加1。对于最终栈不空，再统计栈内左括号不合法的个数即可。

代码如下：

    class Solution {
    public:
        int minAddToMakeValid(string S) {
            stack<int> st;
            int n = S.length();
            int ans = 0;
            for(int i = 0; i < n; i++)
            {
                if(S[i] == '(') st.push(i);
                else if(!st.empty()) st.pop();
                else ans++;
            }
            while(!st.empty()) st.pop(), ans++;
            return ans;
        }
    };

当然做法也很多，之前也见过很多类似的题，比如之前的32题（Hard）。题解给的第二种不用栈的做法，只用两个变量统计左括号和右括号的个数。

题解的代码如下：

    class Solution {
        public int minAddToMakeValid(String S) {
            int ans = 0, bal = 0;
            for (int i = 0; i < S.length(); ++i) {
                bal += S.charAt(i) == '(' ? 1 : -1;
                // It is guaranteed bal >= -1
                if (bal == -1) {
                    ans++;
                    bal++;
                }
            }
            //出循环的时候保证bal>=0，和栈就差不多了，注意体会
            return ans + bal;
        }
    }

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

**题意**：给你一个'0'和'1'组成的字符串，这个字符串是单调递增的当且仅当它包含若干个'0'(可能为0个)，后面跟着若干个'1'(可能为0个)。

我们给你一个字符串S包含'0'和'1'，我们可以进行的操作是把'0'翻转成'1'，或者把'1'翻转成'0'。

让你返回最小的翻转数量，使得S是一个单调递增串。

例子：

Input: "00110"  
Output: 1  
Explanation: We flip the last digit to get 00111.  

Input: "010110"  
Output: 2  
Explanation: We flip to get 011111, or alternatively 000111.  

Input: "00011000"  
Output: 2  
Explanation: We flip to get 00000000.  

数据范围：

1. `1 <= S.length <= 20000`
2. `S`只包含`'0'`和`'1'`这两个字符

**思路**：一开始想着翻转0,1的话，最终要保证字符串前面都是0，后面都是1。那在翻转时一定是把连续的0或者1一起翻转。

所以就先预处理出来(value, cnt)这样pair的数组。然后在这个数组的基础上去dp。因为是序列dp，只考虑把前i段通过翻转变成单调递增子串。然后每一段有两种状态，要么翻要么不翻。然后就能从前一个转移而来。很简单的dp。时间复杂度是O(n)。

代码如下：

    class Solution {
    public:
        int minFlipsMonoIncr(string S) {
            vector<pair<int, int>> vec;
            int n = S.length(), i = 0;
            while(i < n)
            {
                int j = i;
                while(j < n && S[j] == S[i]) j++;
                vec.push_back(make_pair(S[i] - '0', j - i));
                i = j;
            }
            int sz = vec.size();
            vector<vector<int>> dp(sz, vector<int>(2, 0));
            dp[0][1] = vec[0].second;
            for(int i = 1; i < sz; i++)
            {
                dp[i][0] = (vec[i].first == 1) ? min(dp[i - 1][0], dp[i - 1][1]) : dp[i - 1][1];
                dp[i][1] = (vec[i].first == 1) ? dp[i - 1][0] + vec[i].second : min(dp[i - 1][0], dp[i - 1][1]) + vec[i].second;
            }
            return min(dp[sz - 1][0], dp[sz - 1][1]);
        }
    };

题解给的方法更好了，比我这种方法更好写。前缀和求出0~i子串中1的总个数。然后**枚举0的个数**。对于x个0而言，左边区间中`p[x]`个1要翻成0，右边区间内的`N - x - (p[n] - p[x])`个0要翻成1。

注意0的个数可以是0，所以是枚举0的个数。如果枚举分割点的话，就要在-1的位置去枚举，稍微麻烦点，容易出错，同时也容易忘记这个特殊位置。

题解给的Java代码如下：

    class Solution {
        public int minFlipsMonoIncr(String S) {
            int N = S.length();
            int[] P = new int[N + 1];
            for (int i = 0; i < N; ++i)
                P[i+1] = P[i] + (S.charAt(i) == '1' ? 1 : 0);

            int ans = Integer.MAX_VALUE;
            for (int j = 0; j <= N; ++j) {
                ans = Math.min(ans, P[j] + N-j-(P[N]-P[j]));
            }

            return ans;
        }
    }

还有更简单的dp方法：

This is a typical case of DP.  

Let's see the sub-question of DP first.  

Suppose that you have a string `s`, and the solution to the mono increase question is already solved. That is, for string `s`, `counter_flip` flips are required for the string, and there were `counter_one` `'1'`s in the original string `s`.  

Let's see the next step of DP.  

Within the string `s`, a new incoming character, say `ch`, is appended to the original string. The question is that, how should `counter_flip` be updated, based on the sub-question? We should discuss it case by case.  

- When `'1'` comes, no more flip should be applied, since `'1'` is appended to the tail of the original string.  
- When `'0'` comes, things become a little bit complicated. There are two options for us: flip the newly appended `'0'` to `'1'`, after `counter_flip` flips for the original string; or flip `counter_one` `'1'` in the original string to `'0'`. Hence, the result of the next step of DP, in the `'0'` case, is `std::min(counter_flip + 1, counter_one)`;.

Based on these analysis, the solution comes.

    class Solution {
    public:
        int minFlipsMonoIncr(const std::string& S, int counter_one  = 0, int counter_flip = 0) {
            for (auto ch : S) counter_flip = std::min(counter_one += ch - '0', counter_flip + '1' - ch);
            return counter_flip;
        }
    };

If you find the above snippet of code is somewhat difficult to understand, try the below one.

    class Solution {
    public:
        int minFlipsMonoIncr(const std::string& S, int counter_one  = 0, int counter_flip = 0) {
            for (auto ch : S) {
                if (ch == '1') {
                    ++counter_one;
                } else {
                    ++counter_flip;
                }
                counter_flip = std::min(counter_one, counter_flip);
            }
            return counter_flip;
        }
    };

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

## 938. Range Sum of BST

**题意**：找出二叉树中在`[L, R]`区间内的所有节点的value之和。

**思路**：直接递归即可。可以遍历整棵树然后判断每个节点的value的值是否在`[L, R]`的范围内。但是这样没有用到二叉树的性质，通过对当前节点是否在`[L, R]`范围内进行判断，从而可以减少搜索的子树数量，降低时间复杂度。

自己写的代码如下：

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
        int rangeSumBST(TreeNode* root, int L, int R) {
            if(root == NULL) return 0;
            if(root -> val >= L && root -> val <= R) return root -> val + rangeSumBST(root -> left, L, R) + rangeSumBST(root -> right, L, R);
            if(root -> val < L) return rangeSumBST(root -> right, L, R);
            return rangeSumBST(root -> left, L, R);
        }
    };

## 939. Minimum Area Rectangle

**题意**：给你一个xy平面上的点集，每个点都互不相同，确定由这些点组成的最小矩形的面积，矩形的边必须平行于x轴和y轴。

如果不存在任何矩形，则返回0。

例子：  
Input: [[1,1],[1,3],[3,1],[3,3],[2,2]]  
Output: 4  

Input: [[1,1],[1,3],[3,1],[3,3],[4,1],[4,3]]  
Output: 2  

数据范围：

1. `1 <= points.length <= 500`
2. `0 <= points[i][0] <= 40000`
3. `0 <= points[i][1] <= 40000`
4. 所有点都互不相同

**思路**：自己的思路是对于每一个点去暴力枚举与它同行同列的另外两个点，然后判断第四个点是否存在于哈希表中。这样子复杂度实际上还是挺高的，跑了588ms。粗糙分析是O(N^3)的，实际上同行同列的数据虽然是两层循环，但是每一层都达不到N，所以O(N^3)只是一个较为宽松的上界。

要注意在没有矩形的时候，返回0。特判别忘记了。

代码如下：

    class Solution {
    public:
        int minAreaRect(vector<vector<int>>& points) {
            int n = points.size();
            map<int, vector<pair<int, int>> > xvec, yvec;
            set<pair<int, int>> uset;
            for(auto vec: points)
            {
                xvec[vec[0]].push_back(make_pair(vec[0], vec[1]));
                yvec[vec[1]].push_back(make_pair(vec[0], vec[1]));
                uset.insert(make_pair(vec[0], vec[1]));
            }
            int ans = 2e9, flag = 0;
            for(int i = 0; i < n; i++)
            {
                int x = points[i][0], y = points[i][1];
                for(auto px: xvec[x])
                {
                    if(px.second != y)
                    {
                        for(auto py: yvec[y])
                        {
                            if(py.first != x)
                            {
                                if(uset.count(make_pair(py.first, px.second)))
                                    ans = min(ans, abs(py.first - x) * abs(px.second - y)), flag = 1;
                            }
                        }
                    }
                }
            }
            if(!flag) ans = 0;
            return ans;
        }
    };

然后官方题解给了另外两种做法，第一种和我的类似，只不过是枚举一条边的两个点，然后不去枚举第3个点了。而是通过map的形式来实现，把第三个点通过40001*y1 + y2 (y1 < y2，要对这一列的数据先排序，再暴力枚举y1和y2)的形式转成key，对应的x列形成value存在哈希里。然后去搜索。这个复杂度比较复杂，但是个人感觉不是题解上写的O(N^2)。

比较喜欢的是题解的第二种方法，枚举矩形的对角线的两个点。这样就可以得到另外两个点的坐标，然后就可以用哈希表去判断这两个点是否存在。

这个算法时间复杂度是O(N^2)倒是没毛病。

自己按照枚举对角线两个点写的代码：

    class Solution {
    public:
        int minAreaRect(vector<vector<int>>& points) {
            int n = points.size();
            set<pair<int, int>> uset;
            vector<pair<int, int>> pvec;
            for(auto vec: points)
            {
                uset.insert(make_pair(vec[0], vec[1]));
                pvec.push_back(make_pair(vec[0], vec[1]));
            }
            sort(pvec.begin(), pvec.end());
            int ans = 2e9, flag = 0;
            for(int i = 0; i < n; i++)
            {
                int x1 = pvec[i].first, y1 = pvec[i].second;
                for(int j = i + 1; j < n; j++)
                {
                    int x2 = pvec[j].first, y2 = pvec[j].second;
                    if(x1 != x2 && y1 != y2)
                    {
                        if(uset.count(make_pair(x1, y2)) && uset.count(make_pair(x2, y1)))
                            ans = min(ans, abs(x2 - x1) * abs(y2 - y1)), flag = 1;
                    }
                }
            }
            if(!flag) ans = 0;
            return ans;
        }
    };