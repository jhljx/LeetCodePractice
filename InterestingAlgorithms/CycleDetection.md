# Cycle detection problem

从Leetcode的287题开始谈起，注意各种思维之间的转换，这个是非常有趣的。

从整数序列的问题，转换成链表问题，当然也可以看成是**有向图的问题**，从而用图论的方法去解决。视角可以不断地去切换。

需要思考的地方是为什么可以转换成链表问题，在做转换的时候，一定要注意去把握两个问题共性的本质。

**问题定义**  
In computer science, **cycle detection** or **cycle finding** is the algorithmic problem of finding a cycle in **a sequence of iterated function values**.

For any function f that maps a finite set S to itself, and any initial value x0 in S, the sequence of iterated function values

$$x_0,\ x_1=f(x_0),\ x_2=f(x_1),\ \dots,\ x_i=f(x_{i-1}),\ \dots$$

must eventually use the same value twice: there must be some pair of distinct indices `i` and `j` such that `xi = xj`. Once this happens, the sequence must continue periodically, by repeating the same sequence of values from `xi` to `xj − 1`. Cycle detection is the problem of finding `i` and `j`, given `f` and `x0`.



对于Cycle detection的问题，除了Floyd's tortoise and hare algorithm之外，还有Brent's algorithm，这个算法基于exponential search（在二分基础上的改进，先用2的指数形式去判断一个key的边界，然后再二分查找。常数级别的优化）。

## Floyd's tortoise and the hare algorithm

Floy判圈算法，可以判断有限状态机，迭代函数或者链表上是否存在环。实际上和取模的空间很有关系。

如果有限状态机、迭代函数或者链表上存在环，那么在某个环上以不同速度前进的两个指针必定会在某个时刻相遇。这个实际上需要证明。自己可以简单地通过设定两个指针刚进入环上的起始位置x、y，然后去通过取模运算得到(x + v1 * k) % len == (y + v2 * k) % len，即(x - y + (v1 - v2) * k) 是环长度len的倍数。然后就可以选择合适的k，使得上式是len的倍数。

所以说明相遇是一定存在的，简单起见速度可以设置为1和2。
**算法描述**

如果有限状态机、迭代函数或者链表存在环，那么一定存在一个起点可以到达某个环的某处(这个起点也可以在某个环上)。

初始状态下，假设已知某个起点节点为节点S。现设两个指针t和h，将它们均指向S。

接着，同时让t和h往前推进，但是二者的速度不同：t每前进1步，h前进2步。只要二者都可以前进而且没有相遇，就如此保持二者的推进。当h无法前进，即到达某个没有后继的节点时，就可以确定从S出发不会遇到环。反之当t与h再次相遇时，就可以确定从S出发一定会进入某个环，设其为环C。

如果确定了存在某个环，就可以求此环的起点与长度。

上述算法刚判断出存在环C时，显然t和h位于同一节点，设其为节点M。显然，仅需令h不动，而t不断推进，最终又会返回节点M，统计这一次t推进的步数，显然这就是环C的长度。

**为了求出环C的起点，只要令h仍均位于节点M，而令t返回起点节点S，此时h与t之间距为环C长度的整数倍**。

随后，同时让t和h往前推进，且保持二者的速度相同：t每前进1步，h前进1步。持续该过程直至t与h再一次相遇，设此次相遇时位于同一节点P，则节点P即为从节点S出发所到达的环C的第一个节点，即环C的一个起点。

这里为什么是整数倍可以算一下，t走的步数是x，h走的步数是2x。然后它们在环上相遇了。则(2x - x) % len == 0，即它们的步数差是环长度的整数倍。即x是len的整数倍。t指针走到M点的步数正好是x。所以就可以保持t不动了。自己想的时候稍微麻烦点，得出环的长度之后，让两个指针都从头开始走，只不过一个指针先走len长度。然后再两个指针同时走，每次都走1步。这个思想是一样的，都是想让两个指针的步数差是环长度len的整数倍。

**伪代码**

    //令指针t和h均指向起点节点S。
    t := &S
    h := &S
    repeat
        t := t->next
        h := h->next
        //要注意这一判断一般不能省略
        if h is not NULL
            h := h->next
    until t = h or h = NULL
    if h != NULL
        //如果存在环的话
        n := 0
        //求环的长度
        repeat
            t := t->next
            n := n+1
        until t = h
        t := &S
        //求环的一个起点
        while t != h
            t := t->next
            h := h->next
        P := *t

**应用**
对于有限状态机与链表，可以判断从某个起点开始是否会返回到访问过运行过程中的某个状态和节点。

对于迭代函数，可以判断其是否存在周期，以及求出其最小正周期。（以前求的取模条件下迭代函数的周期原来本质思想是这个，注意与链表的判环进行类比，找到其共性的地方）

cycle detection还可以用来找到cryptographic hash function中的哈希冲突。

转换成有向图找环的话，下面这个用栈标记的方法和染色法的三种状态实际上本质是一样的。
[https://www.geeksforgeeks.org/detect-cycle-in-a-graph/](https://www.geeksforgeeks.org/detect-cycle-in-a-graph/)

