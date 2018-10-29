# Problem 1~300 Medium

## 2. Add Two Numbers  

**题意**：给你两个非空的链表代表两个非负的整数。数字的每一位都是逆序存放的，即低位放在链表开头，高位放在链表末尾。

**思路**：直接模拟，类似于高精度的加法，从低位开始算，每次记录余数。然后先把两个数字公共的位数计算完，然后再去考虑多出来的那些位数。同时注意余数的进位。注意细节问题。

## 3. Longest Substring Without Repeating Characters

**题意**： 给你一个字符串，找到没有重复字母的最长子串的长度。

**思路**：

- （自己的做法）：直接遍历然后统计。对于每一个位置的i，使用unordered_set或者unordered_map统计每一个位置开始的不重复子串能延伸多长。复杂度O(n^2)。这样使用unordered_map会有一个向后的跳转，类似kmp的失配跳转，但是区别在于这个没有优化O(n^2)的复杂度，所以还是比较慢。

unordered_set版本的代码：

    class Solution {
    public:
        int lengthOfLongestSubstring(string s) {
            int sz = s.length(), ans = 0, i = 0;
            while(i < sz)
            {
                unordered_set<char> uset;
                int j = i;
                while(j < sz && !uset.count(s[j])) uset.insert(s[j++]);
                ans = max(ans, j - i);
                i++;
            }
            return ans;
        }
    };

unordered_map版本的代码：

    class Solution {
    public:
        int lengthOfLongestSubstring(string s) {
            int sz = s.length(), ans = 0, i = 0;
            while(i < sz)
            {
                unordered_map<char, int> umap;
                int j = i;
                while(j < sz && !umap.count(s[j])) umap[s[j]] = j, j++;
                ans = max(ans, j - i);
                if(j < sz) i = umap[s[j]] + 1;
                else i = j;
            }
            return ans;
        }
    };

- 更优的做法：hashSet + 滑动窗口。将O(n^2)复杂度降低到O(2n) = O(n)。

代码如下：

    public class Solution {  
        public int lengthOfLongestSubstring(String s) {  
            int n = s.length();  
            Set<Character> set = new HashSet<>();  
            int ans = 0, i = 0, j = 0;  
            while (i < n && j < n) {  
                // try to extend the range [i, j]  
                if (!set.contains(s.charAt(j))){  
                    set.add(s.charAt(j++));  
                    ans = Math.max(ans, j - i);  
                }  
                else {  
                    set.remove(s.charAt(i++));  
                }  
            }  
            return ans;  
        }  
    }

- hashMap + 滑动窗口。

代码如下：

    public class Solution {
        public int lengthOfLongestSubstring(String s) {
            int n = s.length(), ans = 0;
            Map<Character, Integer> map = new HashMap<>(); // current index of character
            // try to extend the range [i, j]
            for (int j = 0, i = 0; j < n; j++) {
                if (map.containsKey(s.charAt(j))) {
                    i = Math.max(map.get(s.charAt(j)), i);
                }
                ans = Math.max(ans, j - i + 1);
                map.put(s.charAt(j), j + 1);
            }
            return ans;
        }
    }

## 5. Longest Palindromic Substring

**题意**： 给你一个字符串S，找到S中最长的回文子串。假设S的长度最大为1000。

**思路**： 动态规划。考虑回文串的形成，一个回文串在开头和末尾加上相同的字母即可形成更长的回文串。因而对于回文串而言具有最优子结构的性质。当开头和末尾字母不同时，一个长串整体的最长回文串就等于两个重叠子串部分的最长回文串中更大的那一个。算法复杂度为O(n^2)。

即  
$$dp[l][r] = 1 (l == r)$$
$$dp[l][r] = (s[l] == s[r]) ? 2 : 1 (r = l + 1)$$
$$dp[l][r] = (s[l] == s[r]) ? dp[l + 1][r - 1] + 2 : max(dp[l][r - 1], dp[l + 1][r]) (r >= l + 2)$$

但这样只能得到一个字符串的最大回文子串的长度，求具体哪一个子串是最长的，还需要去记录。

由于这里要求的是最长回文串的具体字符串。所以对于整个字符串而言可以转换为只通过**递推计算**（保证精确定位到回文子串）是回文串的部分，而不去管那些非回文串的部分。

因此代码也可以这么写：

    class Solution {
    public:
        string longestPalindrome(string s) {
            int sz = s.length();
            vector<vector<int>> dp(sz, vector<int>(sz, 0));
            int start = 0, maxlen = 0;
            for(int len = 1; len <= sz; len++)
            {
                for(int i = 0; i + len - 1 < sz; i++)
                {
                    int j = i + len - 1;
                    if(len == 1) dp[i][j] = 1;
                    else if(len == 2 && s[i] == s[j]) dp[i][j] = 2;
                    else if(s[i] == s[j] && dp[i + 1][j - 1]) dp[i][j] = dp[i + 1][j - 1] + 2;
                    if(dp[i][j] > maxlen) {maxlen = dp[i][j], start = i;}
                }
            }
            return s.substr(start, maxlen);
        }
    };

LeetCode提供的官方题解：[https://leetcode.com/problems/longest-palindromic-substring/solution/](https://leetcode.com/problems/longest-palindromic-substring/solution/)

需要注意其中提到的**Algorithm 4：Expand Around Center**这种扩展方法。根据中心向两边扩展，然后通过O(1)的空间存储最优的信息。这个方法也可以看做是Manacher算法的一个前提。Manacher算法可以说是对这个算法的一种优化。

当然，也可以使用Manacher算法来求解最长回文子串。算法复杂度为O(n)。

    class Solution {
    public:
        string longestPalindrome(string s) {
            string nstr = "$#";
            for(int i = 0; i < s.length(); i++) nstr += s[i], nstr += '#';
            int mx = 0, id = 0, maxlen = 0, pos = 0;
            vector<int> p(nstr.length(), 0);
            for(int i = 1; i < nstr.length(); i++)
            {
                if(i < mx)
                    p[i] = min(p[2 * id - i], mx - i);
                else
                    p[i] = 1;
                while(nstr[i + p[i]] == nstr[i - p[i]]) p[i]++;
                if(i + p[i] > mx)
                {
                    mx = i + p[i];
                    id = i;
                }
                if(p[i] - 1 > maxlen)
                {
                    maxlen = p[i] - 1;
                    pos = (i - p[i]) / 2; //这里减maxlen也可以
                }
            }
            return s.substr(pos, maxlen);
        }
    };

## 8. String to Integer(atoi)

**题意**：手动实现atoi函数，将一个string转换成integer。

**思路**：模拟即可。题目有一些特殊情况的限制，没什么算法层面的东西。只要细心即可。

## 29. Divide Two Integers

**题意**：给你两个整数dividend和divisor，实现这两个整数的整数除法，不能使用乘法，除法和取模运算符。

**思路**：由于不能使用乘法，所以相当于是要不断地做减法。反过来就是不断的做加法。因此使用快速乘法，即用二进制的方法来快速做加法。因此**二进制**的思想很重要。这也是**整数的快速幂算法**所带来的启示。此外还要注意使用long long，以及根据题意对INT_MIN和INT_MAX以及溢出情形的特判。

## 33. Search in Rotated Sorted Array

**题意**：假设一个数组按照递增的顺序进行排列，数组在某个位置处进行了旋转。  
比如：[0, 1, 2, 4, 5, 6, 7]可以通过旋转变成[4, 5, 6, 7, 0, 1, 2]。  
给你一个target数字，查找这个数字在数组中的位置，如果能找到返回数字在数组中的下标，找不到返回-1。

**一个重要的前提假设**：数组中不存在重复元素。  
同时要保证算法复杂度在O(log n)级别。

**思路**：

- 虽然是对有序的数组做了一个翻转，但是仍然是部分有序的。所以可以先二分查找出来分界点的位置。然后就能够将数组划分成两端有序的部分，然后分别在这两段里进行查找。总的复杂度还是O(log n)。

使用这种方法先二分分界点，再去二分查找的方法。可以很好地和其他题目对接。比如153题，即为求解该解法的第一步，查找旋转之后最小的那个分界点的数。

代码如下：

    class Solution {
    public:
        int search(vector<int>& nums, int target) {
            int n = nums.size();
            int l = 0, r = n - 1;
            while(l < r)
            {
                int mid = l + (r - l) / 2;
                if(nums[mid] > nums[r]) l = mid + 1;
                else r = mid;
            }
            int rot = r;
            l = 0, r = n - 1;
            while(l <= r)
            {
                int mid = l + (r - l) / 2;
                int id = (rot + mid) % n;
                if(nums[id] == target) return id;
                if(nums[id] < target) l = mid + 1;
                else r = mid - 1;
            }
            return -1;
        }
    };

- 第二种思路是根据二分查找的性质动态维护二分有序的序列。因为二分查找时需要的序列是有序序列（非减序列、或非增序列）。而除了分界点之外，其他部分都是满足有序序列的有序性的，因此仍然可以使用二分查找。

这里需要注意序列中没有重复元素，同时要注意当二分查找缩小到只有两个数字的时候，这是需要注意的边界情况。只有当序列只有两个数字时，left和mid才会指向同一个位置，这时候left~mid是递增的（相等），而不是严格的大于，即`nums[mid] > nums[left]`。所以在写代码时要注意判断有序性是要`>=`符号的，防止出错。

两种情况：
当[left, mid]是递增序列时，由于只旋转了一次，所以分界点必然在右边区间。这时当target位于该区间时，可以正常二分查找。否则切换区间到右边区间。  
另一种情况，当[mid + 1, right]是递增序列时，分界点必然已经在左边区间里了。当target位于该区间时，正常二分。否则切换到左边区间。

代码：

    class Solution {
    public:
        int search(vector<int>& nums, int target) {
            int n = nums.size();
            int l = 0, r = n - 1;
            while(l <= r)
            {
                int mid = l + (r - l) / 2;
                if(nums[mid] == target) return mid;
                if(nums[mid] >= nums[l])
                {
                    if(target >= nums[l] && target < nums[mid]) r = mid - 1;
                    else l = mid + 1;
                }
                else
                {
                    if(target > nums[mid] && target <= nums[r]) l = mid + 1;
                    else r = mid - 1;
                }
            }
            return -1;
        }
    };

前两种方法是按照题意，朴素地将序列分成两段有序的部分。而第三种方法是将整个序列构造成一个有序序列。

假设nums数组为[12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]。由于该序列不是完全有序的，所以可以按照如下的情况做转化：

- 假设target是14，我们只需要调整nums数组成为如下的样子，使用inf填充排除掉的区间，保持整体的有序性。  
    [12, 13, 14, 15, 16, 17, 18, 19, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf]

- 假设target是7，我们调整nums数组成为如下的样子。使用-inf填充排除掉的区间，保持整体的有序性。  
    [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

代码如下：

    class Solution {
    public:
        int search(vector<int>& nums, int target) {
            int n = nums.size();
            int l = 0, r = n - 1;
            while(l <= r)
            {
                int mid = l + (r - l) / 2;
                int val = (nums[mid] >= nums[0]) == (target >= nums[0]) ? nums[mid] : (target >= nums[0]) ? INT_MAX : INT_MIN;
                if(target == val) return mid;
                else if(target > val) l = mid + 1;
                else r = mid - 1;
            }
            return -1;
        }
    };

对于81题(Search in Rotated Sorted Array II)，数组中有重复数字。

考虑重复数字带来的影响，这时候num[left] == nums[mid]的情况不只是在left==mid时出现，在其他情形下也会出现。比如：  
[3, 3, 3, 1, 3, 3, 3, 3, 3], left = 0, mid = 4, right = 8。在nums[left] == nums[mid]的时候，也可能出现左半边不有序的情况。这时候nums[left]==nums[mid]，由于是旋转过的，nums[mid] <= nums[right]，则可以得到三个数都相等。因此对于这种情况要单独处理。排除这种情况，当nums[left] == nums[mid]的时候，左半区间就可以是有序的了。

因此81题的算法是在33题的第二种算法基础上加上特判条件即可。此外，在写的时候注意一开始要判断nums[mid]是否和target相等，其次才是这**三种互斥条件**的判断。。注意判断完特殊情况后是else if（互斥条件）。

相比于33题不存在相同元素的情况，本题的算法平均情况是O(log n)，最坏可以达到O(n)。

代码如下：

    class Solution {
    public:
        bool search(vector<int>& nums, int target) {
            int sz = nums.size();
            int l = 0, r = sz - 1;
            while(l <= r)
            {
                int mid = l + (r - l) / 2;
                if(nums[mid] == target) return true;
                if(nums[mid] == nums[l] && nums[mid] == nums[r]) l++, r--;
                else if(nums[mid] >= nums[l])
                {
                    if(target >= nums[l] && target < nums[mid]) r = mid - 1;
                    else l = mid + 1;
                }
                else
                {
                    if(target > nums[mid] && target <= nums[r]) l = mid + 1;
                    else r = mid - 1;
                }
            }
            return false;
        }
    };

## 46. Permutations

**题意**：给你一个不包含重复数字的集合，返回该集合所有可能的排列。

**思路**：DFS搜索全排列。需要一个记录已经使用了哪些数字的vector。然后回溯。

## 47. Permutations II

**题意**：给你一个包含重复数字的集合，返回该集合所有可能的排列。

**思路**：DFS搜索全排列。先用unordered_map预处理，防止全排列重复。因为一个位置上只能放不相同的数字才不会导致重复。然后回溯。

## 48. Rotate Image

**题意**：给你一个n \* n的2D矩阵，让你把这个顺时针旋转90度。必须是**就地旋转**，而不能新开辟一个二维数组。

**思路**：不能直接覆盖去旋转，只能考虑矩阵元素之间的交换。然后想到先求出矩阵转置，然后再每一行互换对称位置的元素即可。

## 49. Group Anagrams

**题意**： 给你一个string数组，让你求出这个数组中string由相同字母组成的string的数组的集合。
例如：  

    Input: ["eat", "tea", "tan", "ate", "nat", "bat"],  
    Output:  
    [  
    ["ate","eat","tea"],  
    ["nat","tan"],  
    ["bat"]  
    ]  

**思路**：可以考虑hash的方法。但是这个也有讲究。可以是`unordered_map<string, vector<string>>`，但是最后要返回`vector<vector<string>>`。为了节省内存，可以构建这样的hash，即`unordered_map<string, int>`。value表示的是该string对应的在最终返回的`vector<vector<string>>`中的外层vector中的下标。然后动态插入string。还有一个trick是先要将当前遍历的string按char排序。

代码如下：

    class Solution {
    public:
        vector<vector<string>> groupAnagrams(vector<string>& strs) {
            unordered_map<string, int> umap;
            vector<vector<string>> res;
            int idx = 0;
            for(auto str : strs)
            {
                string keystr = str;
                sort(keystr.begin(), keystr.end());
                if(!umap.count(keystr))
                {
                    umap[keystr] = idx++;
                    res.push_back({str});
                }
                else
                {
                    int vidx = umap[keystr];
                    res[vidx].push_back(str);
                }
            }
            return res;
        }
    };

## 50. Pow(x, n)

**题意**: 求解pow(x, n)，其中x是double，n是int范围内。

**思路**：修改原始的整数快速幂的算法，改成double。然后注意n的范围要用long long。对于n为负数的情况，先变成正数情况求解，最后再取倒数。

## 60. Permutation Sequence

**题意**：集合[1,2,3,...,n]包含总共n!个不同的排列。按照顺序列出全部的排列，比如n = 3的排列为：  

1. "123"
2. "132"
3. "213"
4. "231"
5. "312"
6. "321"

给定n和k，返回第k个排列。

**思路**：O(n)的递归查找，根据每一位动态更新当前位数下的排名k。代码如下：

    class Solution {
    public:
        string getPermutation(int n, int k) {
            vector<int> flag(n + 1, 0), cnt(n + 1, 1);
            for(int i = 1; i <= n; i++) cnt[i] = cnt[i - 1] * i;
            string res;
            dfs(n, k, cnt, flag, res);
            return res;
        }
        void dfs(int n, int k, vector<int>& cnt, vector<int>& flag, string& res)
        {
            if(n == 0) return;
            int pos = (k - 1) / cnt[n - 1] + 1, idx = 0;
            for(int i = 1; i <= flag.size(); i++)
            {
                if(!flag[i])
                {
                    idx++;
                    if(idx == pos)
                    {
                        flag[i] = 1;
                        res += (i + '0');
                        dfs(n - 1, (k - 1) % cnt[n - 1] + 1, cnt, flag, res);
                        return;  //为了保证O(n)，这里要return
                    }
                }
            }
        }
    };

## 62. Unique Paths

**题意**：给定一个$m \times n$的矩阵，一个机器人在左上角，每一次它只能向右走，或者向下走。问走到右下角有几种不同的走法？

**思路**：动态规划递推求解。

## 63. Unique Paths II

**题意**：同62题，只不过矩阵中有些地方为障碍物，0表示可以走的位置，1表示障碍物。从左上角走到右下角有多少种走法？

**思路**：因为有障碍物的存在，所以障碍物的地方dp值可以置为0。所以依然可以用动态规划求解。当然也可以BFS搜索。  
**这里第一反应还是要想到动态规划，障碍物的产生造成了什么影响。相当于之前可以走到障碍物的步数现在都清零了。即障碍物的状态不能转移到别的状态。**

## 64. Minimum Path Sum

**题意**：给定一个$m \times n$的矩阵，矩阵元素非负，找到一条从左上角到右下角的路径，保证路径上的元素之和最小。每次只能向下走或者向右走。

**思路**：简单动态规划。

## 81. Search in Rotated Sorted Array II

详见33题。

## 96. Unique Binary Search Trees

**题意**：给定一个正整数n，有多少种结构不同的存储1~n这n个数字的二叉搜索树？

**思路**：卡塔兰数，递推求解。注意思考递归公式。从整体往部分上去想，虽然是递推。但是和动态规划的思考方式是一样的，都是考虑整体与部分的关系。需要找到如何划分状态，并且将一个大的问题分成多个子问题。

这里对于n个节点的二叉搜索树，根节点有n种选取方式，当i为根节点的时候，二叉搜索树左边有i - 1个数字，右边有n - i个数字。所以可以得到递推公式为：

$$F(n) = F(0) \* F(n - 1) + F(1) \* F(n - 2) + ... + F(n - 2) \* F(1) + F(n - 1) \* F(0)$$

关键点：

- **根据根节点进行枚举**。。然后根据二叉搜索树的性质计算。。最后发现递推公式是卡特兰数。。
- 之前做排列的题的启示：**关键的是relative rank比较重要**。。因而枚举谁是节点。。对于剩下的数字。。**只用管个数就可以了，不用管具体是哪些**。。因为总能够分成两组。。（二叉搜索树的特点。。小于i的放i的左边，大于它的放右边）

## 134. Gas Station

**题意**：在一个环形轨道上有n个加油站，第i个加油站的油量是gas[i]。你有一辆汽车，油的存储无限制，从第i个加油站走到第(i + 1)个加油站需要花费的油量为cost[i]。假设你一开始的油量为0。如果你能从一个加油站顺时针走一圈又回到这个加油站，则返回这个加油站的index，否则返回-1。

题目保证了如果存在解，则解唯一。gas数组和cost数组都是非空的而且长度相等，数组中每个元素都是非负数。

**思路**：依次遍历每一个起始的加油站，然后看它能不能走一圈。假设一个起点i，最多能走到j。则可以知道：

$$(gas[i] - cost[i]) + \sum_{k = i + 1} ^ {j} (gas[k] - cost[k]) < 0$$

因为从加油站i可以走到加油站(i + 1)，所以`gas[i] - cost[i] >= 0`。i走不到j + 1，则由上面的公式可知，i到j都走不到j + 1。所以下一次遍历的时候只需要考虑j + 1能否走一圈。这里用到了**贪心的思想**。

然后还需要注意，如何表示一个环形的路径，将原数组复制两份即可。

代码如下：

    class Solution {
    public:
        int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
            int sz = gas.size();
            gas.insert(gas.end(), gas.begin(), gas.end());
            cost.insert(cost.end(), cost.begin(), cost.end());
            int pos = -1;
            for(int i = 0; i < sz;)
            {
                //cout << i << endl;
                int j = i + 1, sum = gas[i], flag = 0;
                for(; j <= i + sz; j++)
                {
                    if(sum < cost[j - 1]) break;
                    if(j == i + sz) {flag = 1, pos = i; break;}
                    sum += gas[j] - cost[j - 1];
                }
                if(flag) break;
                i = j;
            }
            return pos;
        }
    };

## 153. Find Minimum in Rotated Sorted Array

详见33题。

## 187. Repeated DNA Sequences

**题意**：所有的DNA都是由A、G、C、T四个碱基组成，请你找出DNA序列中所有长度为10个字母且出现次数的DNA子串。

**思路**：最朴素的想法肯定是遍历所有的子串，由于子串长度都是10，所以可以想到**滑动窗口方法**。然后使用hash表统计每个字符串的个数。为了降低算法复杂度，少去遍历最终的哈希表，直接动态地将string插入到返回结果的vector中。所以对于之前已经出现过的子串，如果个数置为-1，则不再重复添加。这里的小trick是通过设置-1这样非法的个数来表示**当前子串已经添加到返回结果中**。

代码如下：

    class Solution {
    public:
        vector<string> findRepeatedDnaSequences(string s) {
            unordered_map<string, int> umap;
            int sz = s.length(), len = 10;
            string str = s.substr(0, len);
            umap[str]++;
            vector<string> res;
            for(int i = 1; i + len - 1 < sz; i++)
            {
                str = str.substr(1, len - 1) + s[i + len - 1];
                if(umap.count(str) && umap[str] == -1) continue;
                umap[str]++;
                if(umap[str] > 1)
                {
                    res.push_back(str);
                    umap[str] = -1;
                }
            }
            return res;
        }
    };

## 200. Number of Islands

**题意**：给定一个2d的01矩阵，其中'1'表示陆地，'0'表示水。一个陆地可以向上下左右四个方向延伸（如果这些方向上也是陆地的话）。统计总共的陆地个数。

**思路**：直接使用Flood Fill算法，DFS去求解全1的连通分量。设置一个vis数组即可。只在vis为0的1的位置处进行flood fill，保证算法复杂度为O(n \* m)。

当然用DSU也求全1的连通分量也行。

## 279. Perfect Squares

**题意**： 给你一个正整数n，找到能够由完全平方数(比如1, 4, 9, 16, ...)加和构成n的最小的完全平方数的数量。

比如：n = 12, 12 = 4 + 4 + 4。答案为3。  
n = 13, 13 = 4 + 9。答案为2。

**思路**: 动态规划。

这里列出自己写的两种不同思路的动态规划。

- 先求出所有在n范围内的完全平方数。这里假设我们已经知道了这些完全平方数，而且只能用这些完全平方数。所以一开始初始化的时候这些完全平方数使用完全平方数构成自己的最小个数就是1。然后通过这些完全平方数去更新得到别的数字。

为了优化效率可以先判断n是不是完全平方数，如果是，直接返回1。

代码如下：

    class Solution {
    public:
        int numSquares(int n) {
            int x = sqrt(n);
            if(x * x == n) return 1;
            vector<int> dp(n + 1, 0);
            vector<int> nums;
            for(int i = 1; i <= x; i++) nums.push_back(i * i), dp[i * i] = 1;
            for(int i = 1; i <= n; i++)
            {
                if(dp[i])
                {
                    for(int j = 0; j < nums.size(); j++)
                    {
                        if(i + nums[j] > n) break;
                        int y = i + nums[j];
                        dp[y] = (!dp[y]) ? dp[i] + 1 : min(dp[y], dp[i] + 1);
                    }
                }
            }
            return dp[n];
        }
    };

另一种动态规划类似于硬币问题，看一个数字能否由比它小的一个数字加上一个完全平方数构成，初始化的时候全部初始化成无穷大。

    class Solution {
    public:
        int numSquares(int n) {
            int maxp = sqrt(n), inf = 1e9;
            vector<int> vec, dp(n + 1, inf);
            for(int i = 1; i <= maxp; i++) vec.push_back(i * i);
            dp[0] = 0;
            for(int i = 1; i <= n; i++)
            {
                for(int j = 0; j < vec.size(); j++)
                {
                    if(i - vec[j] < 0) break;
                    if(dp[i - vec[j]] != inf)
                    {
                        dp[i] = min(dp[i], dp[i - vec[j]] + 1);
                    }
                }
            }
            return dp[n];
        }
    };

两种动态规划的时间复杂度都是O(n * sqrt(n))。

通过看别人的解法发现本题也可以使用BFS来搜索，相当于不考虑子问题的重叠性和最优子结构。

也可以使用数学的方法，有定理支撑：每个自然数都可以表示成4个数字的平方之和。主要使用的定理是`Lagrange's four-square theorem`和`Legendre's three-square theorem`这两个定理。

代码如下：

    class Solution
    {  
    private:  
        int is_square(int n)
        {  
            int sqrt_n = (int)(sqrt(n));  
            return (sqrt_n*sqrt_n == n);  
        }
    public:
        // Based on Lagrange's Four Square theorem, there
        // are only 4 possible results: 1, 2, 3, 4.
        int numSquares(int n)
        {  
            // If n is a perfect square, return 1.
            if(is_square(n))
            {
                return 1;  
            }
            // The result is 4 if and only if n can be written in the
            // form of 4^k*(8*m + 7). Please refer to
            // Legendre's three-square theorem.
            while ((n & 3) == 0) // n%4 == 0  
            {
                n >>= 2;  
            }
            if ((n & 7) == 7) // n%8 == 7
            {
                return 4;
            }
            // Check whether 2 is the result.
            int sqrt_n = (int)(sqrt(n));
            for(int i = 1; i <= sqrt_n; i++)
            {  
                if (is_square(n - i*i))
                {
                    return 2;  
                }
            }
            return 3;  
        }  
    };
