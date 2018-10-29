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

**思路**：题目给的定义就符合递归的定义。