# Problem 1~100 Medium

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

