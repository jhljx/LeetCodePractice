

https://leetcode.com/problems/create-maximum-number/discuss/77285/Share-my-greedy-solution

https://leetcode.com/submissions/detail/180825080/

/*
Your logic: 
Sample problem: 
[3, 4, 6, 5]
[9, 1, 2, 5, 8, 3]

For array element a[i], we build count of elements smaller than current element in the remaining part of array from i to n-1. 

[3, 4, 6, 5] => [0, 0, 1, 0]
[9, 1, 2, 5, 8, 3] => [5, 0, 0, 1, 1, 0]

We add the "remaining"(see how remaining changes as we pick elements) numbers in the other array to this array: 

[3, 4, 6, 5] => [6, 6, 7, 6]
[9, 1, 2, 5, 8, 3] => [9, 4, 4, 5, 5, 4]

-> Iteration 1
From these numbers we pick the max i from the elements with Count(i) >= (k-1)= 4

Which is 9. So first number is 9. 

We update the count arrays with this change and mark all elements from the picked element to the start as -1: 

[3, 4, 6, 5] => [5, 5, 6, 5]
[9, 1, 2, 5, 8, 3] => [-1, 4, 4, 5, 5, 4]

-> Iteration 2

From these numbers we pick the max i from the elements with Count(i) >= (k-2) = 3
Pick 8
[3, 4, 6, 5] => [1, 1, 2, 1]
[9, 1, 2, 5, 8, 3] => [-1, -1, -1, -1, -1, 4]

-> Iteration 3

From these numbers we pick the max i from the elements with Count(i) >= (k-3) = 2
Pick 6
[3, 4, 6, 5] => [-1, -1, -1, 1]
[9, 1, 2, 5, 8, 3] => [-1, -1, -1, -1, -1, 1]

-> Iteration 4

From these numbers we pick the max i from the elements with Count(i) >= (k-4) = 1

Pick 5
[3, 4, 6, 5] => [-1, -1, -1, -1]
[9, 1, 2, 5, 8, 3] => [-1, -1, -1, -1, -1, 0]

-> Iteration 5

From these numbers we pick the max i from the elements with Count(i) >= (k-5) = 0
Pick 3. 

So total number is 98653. 


If there are no i such that Count(i) >= (k-iteration), we might get away with picking the maximum of the (k-iteration)th element from the end of the one of the arrays which are already sorted. This will look similar to merging to sorted arrays. 


Complexity: 
First compute of count is O(n^2) + O(m^2). 
After that there are k steps. Each step's complexity is O(n + m). So total k \*O(m+n). 


Total complexity: O(n^2) + O(m^2) + k\*O(m+n). 

- search method
https://leetcode.com/submissions/detail/180825118/
