## 用两个栈实现队列

用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

> 思路：一直往第一个栈里压入，取出时，如果第二个栈为空，那么 第一个出栈压入第二个栈，最后弹出第二个栈的栈顶

```java
import java.util.Stack;

public class Solution {
  Stack<Integer> stack1 = new Stack<Integer>();
  Stack<Integer> stack2 = new Stack<Integer>();

  public void push(int node) {
    stack1.add(node);
  }

  public int pop() {
    while (stack2.isEmpty()) {
      while (!stack1.isEmpty()) {
        stack2.push(stack1.pop());
      }
    }
    return stack2.pop();
  }
}
```

## 变态跳台阶

一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

> 思路：前面所有情况 + 一步到顶的情况

```javascript
function jumpFloorII(number) {
  if (number <= 1) return number;
  let result = [0, 1]
  for (let i = 2; i <= number; i++) {
    let tmp = 0;
    for (let j = 0; j < result.length; j++) {
      tmp += result[j]
    }
    result[i] = tmp + 1
  }
  return result.pop()
}
```

## 矩形覆盖

我们可以用2\*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2\*1的小矩形无重叠地覆盖一个2\*n的大矩形，总共有多少种方法？

> 思路：result[i] = result[i - 1] + result[i - 2]; 实际上就是上一次加一个竖着的 + 上上次加两个横着的

```javascript
function rectCover(number) {
  if (number <= 2) return number;
  let f1 = 1, f2 = 2, tmp;
  for (let i = 2; i < number; i++) {
    tmp = f2;
    f2 = f1 + f2
    f1 = tmp;
  }
  return f2
}
```

## 二进制中1的个数

输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

> 思路：经典位运算，减一就相当于把第一个1及其后面的都取反，与元数据与一下，就能得到去掉第一个1的数。
> 如 10100 - 1 = 10011， 10011 & 10100 = 10000

```javascript
function NumberOf1(n) {
  let i = 0;
  while (n !== 0) {
    n = n & (n - 1)
    i++
  }
  return i;
}
```

## 调整数组顺序使奇数位于偶数前面

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

> 思路：找到第一个偶数，在找到第一个偶数之后的奇数，插入到第一个偶数的位置上，并将偶数及其后面的数顺延一位

```javascript
function reOrderArray(array) {
  let len = array.length, i = 0, j, i1;
  for (; i < len; i++) {
    if (array[i] % 2 === 0) break;
  }
  for (j = i + 1; j < len; j++) {
    if (array[j] % 2 === 1) {
      let tmp = array[i]
      array[i] = array[j];
      i1 = i
      while (++i <= j) {
        let tmp2 = array[i]
        array[i] = tmp
        tmp = tmp2
      }
      i = i1 + 1
    }
  }
  return array
}
```

## 链表中倒数第 K 个节点

输入一个链表，输出该链表中倒数第k个结点。

> 思路：注意越界

```java
/*
public class ListNode {
    int val;
    ListNode next = null;

    ListNode(int val) {
        this.val = val;
    }
}*/
public class Solution {
    public ListNode FindKthToTail(ListNode head,int k) {
        if (head == null || k == 0) return null;
        ListNode node = head;
        while (node.next != null) {
            if (--k <= 0) head = head.next;
            node = node.next;
        }
        return k > 1 ? null : head;
    }
}
```

## 反转链表

输入一个链表，反转链表后，输出新链表的表头。

```javascript
/*function ListNode(x){
    this.val = x;
    this.next = null;
}*/
function ReverseList(pHead) {
  let tmp = null, x = pHead, y = null;
  while(x !== null) {
    tmp = x.next;
    x.next = y;
    y = x;
    x = tmp
  }
  return y;
}
```

## 合并两个排序的链表

输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

```javascript
/*function ListNode(x){
    this.val = x;
    this.next = null;
}*/
function Merge(pHead1, pHead2) {
  let newHead = new ListNode(null)
      tmpHead = newHead;
  while (pHead1 !== null && pHead2 !== null) {
    if (pHead1.val > pHead2.val) {
      tmpHead.next = pHead2;
      pHead2 = pHead2.next;
    } else {
      tmpHead.next = pHead1;
      pHead1 = pHead1.next;
    }
    tmpHead = tmpHead.next;
  }
  if (pHead1 === null) tmpHead.next = pHead2
  else tmpHead.next = pHead1
  
  return newHead.next;
}
```

## 树的子结构

输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

```javascript
/* function TreeNode(x) {
    this.val = x;
    this.left = null;
    this.right = null;
} */
function HasSubtree(pRoot1, pRoot2) {
  if (pRoot2 === null) return false
  let q = []
  while(pRoot1 !== null || q.length > 0) {
    while (pRoot1 !== null) {
      q.push(pRoot1)
      pRoot1 = pRoot1.left
    }
    let t = q.pop()
    if (handler(t, pRoot2))
      return true
    pRoot1 = t.right
  }
  return false
}

function handler(node1, node2) {
  if (node1 === null) {
    return node2 === null
  } else if (node2 === null) {
      return true
  }
  return node1.val === node2.val &&
         handler(node1.left, node2.left) &&
         handler(node1.right, node2.right)
}
```

## 二叉树的镜像

操作给定的二叉树，将其变换为源二叉树的镜像。

```javascript
/* function TreeNode(x) {
    this.val = x;
    this.left = null;
    this.right = null;
} */
function Mirror(root) {
  if (root === null) return
  let l = root.left,
      r = root.right
  root.left = Mirror(r)
  root.right = Mirror(l)
  return root;
}
```