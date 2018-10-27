# 2. Add Two Numbers  

**题意**：给你两个非空的链表代表两个非负的整数。数字的每一位都是逆序存放的，即低位放在链表开头，高位放在链表末尾。

**思路**：直接模拟，类似于高精度的加法，从低位开始算，每次记录余数。然后先把两个数字公共的位数计算完，然后再去考虑多出来的那些位数。同时注意余数的进位。注意细节问题。

# 3. Longest Substring Without Repeating Characters

**题意**： 给你一个字符串，找到没有重复字母的最长子串的长度。

**思路**： 
- （自己的做法）：直接遍历然后统计。对于每一个位置的i，使用unordered_set或者unordered_map统计每一个位置开始的不重复子串能延伸多长。复杂度O(n^2)。
- 更优的做法：hashSet + 滑动窗口。将O(n^2)复杂度降低到O(2n) = O(n)。

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