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

## 顺时针打印矩阵

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

> 思路：直接圈定输出位置

```javascript
/**
 * 
 * @param {number[][]} matrix 
 */
function printMatrix(matrix) {
  let row = matrix.length,
      col = (matrix[0] || []).length;
  let result = []
  let l = 0, t = 0, r = col - 1, b = row - 1;
  while(l <= r && t <= b) {
    for (let i = l; i <= r; i++) result.push(matrix[t][i])
    for (let i = t + 1; i <= b; i++) result.push(matrix[i][r])

    for (let i = r - 1; i >= l && t < b; i--) result.push(matrix[b][i])
    for (let i = b - 1; i > t && l < r; i--) result.push(matrix[i][l])
    l++, t++, r--, b--;
  }
  return result
}
```

## 包含 min 函数的栈

定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。

> 思路：创建一个辅助栈，每次 push 的时候，判断该值与辅助栈栈顶的大小，如果小于辅助栈栈顶，入栈。
> 出栈时都要出栈。

```java
import java.util.Stack;

public class Solution {
    private Stack<Integer> a = new Stack<>();
    private Stack<Integer> b = new Stack<>();
    
    public void push(int node) {
        a.add(node);
        if (b.isEmpty() || b.peek() > node) {
          b.add(node);
        } else {
          b.add(b.peek());
        }
    }
    
    public void pop() {
      a.pop();
      b.pop();
    }
    
    public int top() {
      return a.peek();
    }
    
    public int min() {
      return b.peek();
    }
}
```

## 从上往下打印二叉树

从上往下打印出二叉树的每个节点，同层节点从左至右打印。

```javascript
/* function TreeNode(x) {
    this.val = x;
    this.left = null;
    this.right = null;
} */
function PrintFromTopToBottom(root) {
  if (root === null) return [];
  let queue = [root],
      num = 1,
      result = [];
  while(queue.length > 0) {
    let tmpNum = 0, tmpArr = [];
    while(num-- > 0) {
      let tmpNode = queue.shift()
      tmpArr.push(tmpNode.val)
      if (tmpNode.left) {
        tmpNum++;
        queue.push(tmpNode.left)
      }
      if (tmpNode.right) {
        tmpNum++;
        queue.push(tmpNode.right)
      }
    }
    num = tmpNum;
    result.push(tmpArr)
  } 
  return result;
}
```

## 二叉搜索树的后序遍历序列

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

> 思路：取最后一位，遍历前面数组，找到第一个大于最后一位的值，该位置至最后一位，不能出现小于最后一位的值，递归得出结果。

```javascript
function VerifySquenceOfBST(sequence) {
  if (sequence.length === 0) return false;
  return handler(sequence, 0, sequence.length - 1)
}

function handler(sequence, start, end) {
  if (start >= end) return true;
  let flag = false, endNum = sequence[end], mid;
  for (let i = start; i < end; i++) {
    if (sequence[i] > endNum) {
      if (!flag) mid = i
      flag = true
    } else if (flag) {
      return false
    }
  }
  if (!flag) mid = end;
  return handler(sequence, start, mid - 1) && handler(sequence, mid, end - 1)
}
```

## 二叉树中和为某一路径的值

输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)

```javascript
/* function TreeNode(x) {
    this.val = x;
    this.left = null;
    this.right = null;
} */
function FindPath(root, expectNumber) {
  if (root === null) return [];
  let result = []
  handler(root, expectNumber, result, [])
  return result.sort((a, b) => b.length - a.length)
}

function handler(node, expectNumber, result, currArr) {
  if (expectNumber < 0 || node === null) {
    return
  }
  currArr.push(node.val)
  expectNumber -= node.val
  if (expectNumber === 0) {
    if (node.left === null && node.right === null)
      result.push([...currArr])
  } else {
    handler(node.left, expectNumber, result, currArr)
    handler(node.right, expectNumber, result, currArr)
  }
  currArr.pop()
}
```

## 复杂链表的复制

输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

> 思路：由于可能链表的 label 不唯一，使用 map 方法的话会导致结果不一致，所以使用直接在老节点后面跟一个其复制节点，新节点的 random 实际上就是 old.random.next，最后取出老节点

```javascript
/*function RandomListNode(x){
    this.label = x;
    this.next = null;
    this.random = null;
}*/
function Clone(pHead) {
  let tHead = pHead, nHead = null, nHead1 = null;
  // 创建节点跟在原节点后
  while (tHead !== null) {
    let tN = tHead.next
    let nN = new RandomListNode(tHead.label)
    tHead.next = nN
    nN.next = tN
    tHead = tN
  }
  // 添加循环节点
  tHead = pHead
  while(tHead !== null) {
    if (tHead.random) {
      tHead.next.random = tHead.random.next
    }
    tHead = tHead.next.next
  }
  // 取出新节点
  tHead = pHead
  while (tHead !== null) {
    if (nHead == null) {
      nHead = tHead.next
      nHead1 = nHead
    } else {
      nHead1.next = tHead.next
      nHead1 = nHead1.next
    }
    tHead = tHead.next.next
  }
  return nHead;
}
```

## 二叉搜索树与双向链表

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

```javascript
/* function TreeNode(x) {
    this.val = x;
    this.left = null;
    this.right = null;
} */
function Convert(pRootOfTree) {
  if (pRootOfTree === null ||
      pRootOfTree.next === null)
    return pRootOfTree
  let queue = [],
      pre = null, head = null;
  while (pRootOfTree !== null || queue.length > 0) {
    while (pRootOfTree !== null) {
      queue.push(pRootOfTree)
      pRootOfTree = pRootOfTree.left
    }
    let t = queue.pop()
    if (pre === null) {
      head = t
    } else {
      pre.right = t
      t.left = pre
    }
    pre = t
    pRootOfTree = t.right
  }
  return head;
}
```

## 数组中出现次数超过一半的数字

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

> 思路: 摩尔投票法只能选出可能超过一半的数字，是否真的超过，需要再次遍历计算出现次数

```javascript
function MoreThanHalfNum_Solution(numbers) {
  let len = numbers.length;
  if (len === 0) return false;
  let flag = null, num = 0;
  for (let i = 0; i < len; i++) {
    if (flag === numbers[i]) num++
    else if (num === 0) flag = numbers[i], num = 1
    else num--
  }
  num = 0
  for (let i = 0; i < len; i++) {
    if (numbers[i] === flag) num++
  }
  return num > numbers.length / 2 ? flag : 0
}
```

## 最小的K个数

输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

> 思路：设置一个数组，维持长度等于K

```javascript
function GetLeastNumbers_Solution(input, k) {
	let result = [], len = input.length;
	if (len === 0 || k === 0 || len < k) return []
	result.push(input[0]) 
	for (let i = 1; i < len; i++) {
		for (let j = 0; j <= result.length; j++) {
			if (j === result.length || result[j] > input[i]) {
				result.splice(j, 0, input[i])
				if (result.length > k) result.pop()
				break
			}
		}
	}
	
	return result
}
```

## 连续子数组的最大和

例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和。

```javascript
function FindGreatestSumOfSubArray(array) {
	let sum = 0, max = -0XFFFFFFFF, len = array.length;
	for (let i = 0; i < len; i++) {
		sum += array[i]
		max = Math.max(sum, max)
		if (sum < 0)
			sum = 0
	}
	return max
}
```

## 从1到n整数中1出现的次数

求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

> 思路：规律
> 个位： 我们知道在个位数上，1会每隔10出现一次，例如1、11、21等等，我们发现以10为一个阶梯的话，每一个完整的阶梯里面都有一个1，所以：n/10 * 1+(n%10!=0 ? 1 : 0)
> 十位：十位数上出现1的情况应该是10-19，依然沿用分析个位数时候的阶梯理论，我们知道10-19这组数，每隔100出现一次，这次我们的阶梯是100，我们考虑如果露出来的数大于19，那么直接算10个1就行了，因为10-19肯定会出现；如果小于10，那么肯定不会出现十位数的1；如果在10-19之间的，我们计算结果应该是k - 10 + 1。
> 设 k = n % 10，有 (n / 100) * 10 + (if(k > 19) 10 else if(k < 10) 0 else k - 10 + 1)
> 百位：在百位，100-199都会出现百位1，一共出现100次，阶梯间隔为1000，100-199这组数，每隔1000就会出现一次。 k = n % 100，有 (n / 1000) * 100 + (if(k >199) 100 else if(k < 100) 0 else k - 100 + 1)

```javascript
function NumberOf1Between1AndN_Solution(n) {
	if (n <= 0) return 0
	let count = 0;
	for (let i = 1; i <= n; i *= 10) {
		let t = i * 10,
			k = n % t
		count += (n / t | 0) * i
		count += (k > 2 * i - 1) ? i : k < i ? 0 : (k - i + 1)
	}
	return count
}
```

## 把整数排成最小的数

输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

> 思路：冒泡

```javascript
function PrintMinNumber(numbers) {
	numbers = numbers.map(String)
	let len = numbers.length
	for (let i = 0; i < len - 1; i++) {
		for (let j = 0; j < len - i - 1; j++) {
			if (Number(numbers[j] + numbers[j + 1]) > Number(numbers[j + 1] + numbers[j])) {
				let t = numbers[j]
				numbers[j] = numbers[j + 1]
				numbers[j + 1] = t
			}
		}
	}
	return numbers.join("")
}
```

## 丑数

把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

> 思路：一个丑数的因子只有2,3,5，那么丑数p = 2 ^ x * 3 ^ y * 5 ^ z，换句话说一个丑数一定由另一个丑数乘以2或者乘以3或者乘以5得到。
> 维护三个队列，分别为之前的丑数 * 2，* 3，* 5；然后选出他们当中最小的那个放入队列
> 真正实现是不需要维护3个队列，只需要记住他们走到了主队列中的哪一步就好了

```javascript
function GetUglyNumber_Solution(index) {
	// 前六个数都是丑数
	if (index <= 6) return index;
	let p2 = 0, p3 = 0, p5 = 0, newNum = 1;
	let result = [newNum];
	while(result.length < index) {
		newNum = Math.min(result[p2] * 2, result[p3] * 3, result[p5] * 5)
		if (result[p2] * 2 === newNum) p2++;
		if (result[p3] * 3 === newNum) p3++;
		if (result[p5] * 5 === newNum) p5++;
		result.push(newNum)
	}
	return result.pop()
}
```

## 第一个只出现一次的字符

在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）

> 思路：map

```javascript
function FirstNotRepeatingChar(str)　{
  let obj = {}, obj1 = {}
  for (let i = 0; i < str.length; i++) {
      if (obj1[str[i]]) continue
      if (obj[str[i]] !== undefined) {
        delete obj[str[i]]
        obj1[str[i]] = true
      } else
        obj[str[i]] = i
  }
  return Object.keys(obj).length > 0 ? obj[Object.keys(obj)[0]] : -1
}
```

## 数组中的逆序对

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007

> 思路：将数组分为两份（直到只有一个数据项），合并时，从后往前合并，如果arr[i] > arr[j], 则 arr[mid + 1] ~ arr[j] 都是小于 arr[i] 的 count+= j - (mid+1) +1
> 嗯，是的，就是归并排序……

```JavaScript
function InversePairs(data) {
  let len = data.length;
  if (len < 2) return 0;
  let copy = [];
  for (let i = 0; i < len; i++) copy[i] = data[i]
  return handler(data, copy, 0, len - 1)
}

function handler(data, copy, start, end) {
  if (start == end) {
    copy[start] = data[start]
    return 0;
  }

  let mid = (start + end) / 2 | 0
  let leftCount = handler(data, copy, start, mid)
  let rightCount = handler(data, copy, mid + 1, end)

  let i = mid, j = end,
      copyIdx = end
  let count = 0;
  while (i >= start && j >= mid + 1) {
    if (data[i] > data[j]) {
      count += j - mid
      copy[copyIdx--] = data[i--]
      if (count >= 1000000007)
        count %= 1000000007
    } else {
      copy[copyIdx--] = data[j--]
    }
  }
  while (i >= start) copy[copyIdx--] = data[i--]
  while (j >= mid + 1) copy[copyIdx--] = data[j--]
  for (let i = start; i <= end; i++) data[i] = copy[i]
  return (leftCount + rightCount + count) % 1000000007
}
```

## 两个链表的第一个公共交点

输入两个链表，找出它们的第一个公共结点。

> 思路：首先如果他们有公共交点，则后面的一定也是相交的
> 方案一：map，缺点是如果有重复值就无效
> 方案二：遍历两个链表，第一遍历完后，就连接到另一个头上

```javascript
/*function ListNode(x){
    this.val = x;
    this.next = null;
}*/
// 方案一
function FindFirstCommonNode(pHead1, pHead2) {
  if (pHead1 === null || pHead2 === null) return null;
  let obj = {}
  while (pHead1 !== null) {
    if (obj[pHead1.val]) obj[pHead1.val].push(pHead1)
    else obj[pHead1.val] = [pHead1]
    pHead1 = pHead1.next
  }
  while (pHead2 !== null) {
    if (obj[pHead2.val]) return obj[pHead2.val][0]
    pHead2 = pHead2.next
  }
  return null
}

// 方案二
function FindFirstCommonNode(pHead1, pHead2) {
  if (pHead1 === null || pHead2 === null) return null;
  let pHead1_1 = pHead1, pHead2_1 = pHead2
  while (pHead1 !== pHead2) {
    pHead1 = pHead1 === null ? pHead2_1 : pHead1.next
    pHead2 = pHead2 === null ? pHead1_1 : pHead2.next
  }
  return pHead1
}
```

## 统计一个数字在排序数组中出现的次数。

统计一个数字在排序数组中出现的次数。

> 思路：
> 方案一：一遍遍历，时间复杂度 O(n)
> 方案二：因为是排序数组，直接上二分法查找左右边界，时间复杂度 O(2logn)

```javascript
function GetNumberOfK(data, k) {
  let len = data.length
  if (len < 1) return 0
  // 求左节点
  let l = 0, r = 0, end = len - 1;
  while (l < end) {
    let mid = (l + end) >> 1
    if (data[mid] > k) end = mid - 1
    else if (data[mid] < k) l = mid + 1
    else end = mid
  }
  // 求右节点
  end = len - 1
  while (r < end) {
    let mid = (r + end + 1) >> 1
    if (data[mid] > k) end = mid - 1
    else if (data[mid] < k) r = mid + 1
    else r = mid
  }

  return data[l] !== k ? 0 : r - l + 1
}
```

## 二叉树的深度

输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

> 思路：递归，选择左子树和右子树最大的进行相加

```javascript
/* function TreeNode(x) {
    this.val = x;
    this.left = null;
    this.right = null;
} */
function TreeDepth(pRoot) {
  if (pRoot === null) return 0;
  return Math.max(TreeDepth(pRoot.left), TreeDepth(pRoot.right)) + 1
}
```

## 平衡二叉树

给定一个二叉树，判断该二叉树是否是平衡二叉树

> 思路：平衡二叉树是指其左子树深度和右子树深度不超过1，其左子树和右子树也是平衡二叉树

```javascript
/* function TreeNode(x) {
    this.val = x;
    this.left = null;
    this.right = null;
} */
function IsBalanced_Solution(pRoot) {
  return handler(pRoot) > -1
}

function handler(node) {
  if (node === null) return 0;
  let ld = handler(node.left)
  if (ld === -1) return -1
  let rd = handler(node.right)
  if (rd === -1 || Math.abs(ld - rd) > 1) return -1
  return Math.max(ld, rd) + 1
}
```

## 数组中只出现一次的数

一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

> 思路：还是异或。异或后得出一个数，该数肯定是 r1 ^ r2 的值，该值的 1 肯定是他俩的值不同，找出其他的同样位置的 1，该数组中肯定包含r1 或 r2，因为这个 1 说明了他们的差异，异或该数组得出该值，再异或之前的值得出另一个
> tmp = r1 ^ r2，  那么 r2 = r1 ^ tmp

```javascript
/**
 * 
 * @param {number[]} array 
 */
function FindNumsAppearOnce(array) {
  let len = array.length;
  if (len < 2) return []
  // 求出异或结果
  let tmp = array.reduce((a, b) => a ^ b);

  if (tmp === 0) return []

  let k = find1(tmp), 
      // 找出同样位置的数
      tmpArr = array.filter(val => find1(val) === k),
      // 其中肯定只包含一个出现一次的数
      r1 = tmpArr.reduce((a, b) => a ^ b),
      // tmp = r1 ^ r2, r2 = tmp ^ r1
      r2 = r1 ^ tmp

  return [r1, r2]
}

// 找出结果的第一个 1 的位置
function find1(num) {
  let k = 0;
  while(num !== 0) {
    if (num & 1 === 1) return k
    num >>= 1
    k++
  }
  return -1
}
```

## 和为 S 的连续正序列

> 思路：滑动窗口法

```javascript
function FindContinuousSequence(sum) {
  let tmpSum = 0, tmp = [], result = [];
  for (let i = 1; i <= sum;) {
    if (tmpSum < sum) {
      tmpSum += i
      tmp.push(i++)
    } else if (tmpSum > sum) {
      tmpSum -= tmp.shift()
    } else {
      result.push([...tmp])
      tmpSum -= tmp.shift()
    }
  }
  return result
}
```

## 和为 S 的两个数字

输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

> 思路：双指针

```java
import java.util.ArrayList;
public class Solution {
  public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {
    int len = array.length;
    ArrayList<Integer> result = new ArrayList<>();
    if (len < 1) return result;
    int l = 0, r = len - 1, min = Integer.MAX_VALUE;
    while (l < r) {
      int lVal = array[l], rVal = array[r];
      if (lVal + rVal > sum) r--;
      else if (lVal + rVal < sum) l++;
      else {
        if (min > lVal * rVal) {
          min = lVal * rVal;
          result.clear();
          result.add(lVal);
          result.add(rVal);
        }
        l++;
        r--;
      }
    }
    return result;
  }
}
```

## 左旋字符串

汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”

> 思路：从 n 开始，到 len + n 结束

```javascript
function LeftRotateString(str, n) {
  if (!str) return "";
  let len = str.length;
  if ((n %= len) === 0) return str;
  let result = ""
  for (let i = n; i < len + n; i++) {
    if (i < len) result += str[i]
    else result += str[i - len]
  }
  return result
}
```

## 扑克牌顺子

大小王可以为任意值

> 思路：主要是填充值，得出 大小王的个数，顺子的差值为1，用大小王填充缺的值，如果最后大小王 >= 0，则一定是顺子。

```javascript
function IsContinuous(numbers) {
  let len = numbers.length;
  if (len < 5) return false;
  numbers = numbers.sort((a, b) => a - b)
  let king = 0, prev = -1;
  for (let i = 0; i < numbers.length; i++) {
    if (numbers[i] === 0){
      king ++
      continue
    }
    if (prev === -1) {
      prev = numbers[i]
      continue
    }
    let t = numbers[i] - prev
    if (t === 0) return false
    king = king - (t - 1)
    prev = numbers[i]
  }
  return king >= 0
}
```

## 孩子们的游戏（圆圈中最后剩下的数）

首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,

> 思路：循环链表

```javascript
function LastRemaining_Solution(n, m) {
  if (n <= 0 || m <= 0) return -1
  // 先组个链表
  let head = new LinkedNode(null),
      tmpHead = head
  for (let i = 0; i < n; i++) {
    tmpHead.next = new LinkedNode(i)
    tmpHead = tmpHead.next
  }
  tmpHead.next = head.next // 组成链表环

  let c = 0;
  head = head.next;
  while (head.next !== head) {
    if (c++ === m - 2) {
      head.next = head.next.next
      c = 0
    }
    head = head.next
  }
  return head.val
}

function LinkedNode(val) {
  this.val = val
  this.next = null
}
```

## 求 1+2+3+...+n

求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

> 思路: 主要还是用来做递归的终止条件。
> 方法一：try...catch 大法，取数组或除以 0，即越界终止递归
> 方法二：&& ，如果第一个为 false，则不再进行后面的判断，把递归写后面。

```javascript
function Sum_Solution(n) {
  let a = n
  a !== 0 && (a += Sum_Solution(n - 1))
  return a
}
```

## 不用加减乘除做除法

写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

> 思路：^ 得出个位， & << 1 可以得出十位，递归 & === 0

```javascript
function Add(num1, num2){
  if (num2 === 0) return num1
  let x = num1 ^ num2, y = (num1 & num2) << 1
  return Add(x, y)
}
```

## 数组中重复的数字

在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字

> 思路：用辅助数组，因为是0 ~ n-1, 所以不用担心数组过长，如果 arr\[number\[i]] === true，说明之前出现过.
> 另外，有人说可以在原数组上 + n，如果这个再次索引到的值大于 n，则这个值之前出现过，不过缺点是如果数组长度大于 int 的一般，会溢出

```javascript
function duplicate(numbers, duplication) {
  let len = numbers.length;
  let tmp = [];
  for (let i = 0; i < len; i++) {
    if (tmp[numbers[i]]) {
      duplication[0] = numbers[i]
      return true
    }
    tmp[numbers[i]] = true
  }
  return false
}
```

## 构建乘积数组

给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。就是不含当前位置的乘积。

> 思路：定义两个值，一个从左往右乘，一个从右往左乘

```javascript
function multiply(array) {
  let len = array.length;
  if (len < 1) return [];
  let result = array.map(() => 1)
  let l = 1, r = 1;
  for (let i = 0; i < len; i++) {
    result[i] = result[i] * l
    l *= array[i]

    result[len - i - 1] = result[len - i - 1] * r
    r *= array[len - i - 1]
  }
  return result
}
```

## 链表中的入口节点

给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。

> 思路：快慢指针，定义慢指针 p1 已走环外距离 a，环内距离 b，环内未走为 c，则有快指针已走 2*(a + b) = a + b + c + b，即 a = c，所以只需另一个慢指针和 p1 相遇即为入口点

```javascript
/*function ListNode(x){
    this.val = x;
    this.next = null;
}*/
function EntryNodeOfLoop(pHead) {
  // write code here
  if (pHead === null) return null;
  let p1 = pHead, p2 = pHead;
  while (p2.next !== null) {
    p1 = p1.next;
    p2 = p2.next.next;
    if (p1 === p2) break;
  }
  if (p2.next === null) return null;

  p2 = pHead
  while (p1 !== p2) {
    p1 = p1.next
    p2 = p2.next
  }
  return p1
}
```

## 删除链表中重复的节点

在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

> 思路：双层循环，第二层循环和第一层的值对比，注意如果有重复值需要把外层的删除

```javascript
function ListNode(x){
    this.val = x;
    this.next = null;
}
function deleteDuplication(pHead) {
  if (pHead === null || pHead.next === null)
    return pHead
  let nHead = new ListNode(0)
  nHead.next = pHead
  let p1 = nHead;
  while (p1.next !== null) {
    let flag = false, p2 = p1.next
    while (p2.next !== null) {
      if (p2.val === p2.next.val) {
        flag = true
        p2.next = p2.next.next
      } else break
    }
    if (flag) p1.next = p1.next.next
    else p1 = p2
  }
  return nHead.next
}
```

## 二叉树的第一个节点

给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

> 思路：如果该节点有右子树，那么输出右子树的正常中序遍历
> 如果节点无右子树:（1）如果是根节点，则输出null,（2）如果非跟节点，如果该节点是其父节点的左孩子，则返回父节点；否则继续向上遍历其父节点的父节点，

```javascript
/*function TreeLinkNode(x){
    this.val = x;
    this.left = null;
    this.right = null;
    this.next = null;
}*/
function GetNext(pNode) {
  if (pNode.right !== null) {
    let t = pNode.right
    while (t.left !== null) t = t.left
    return t
  } else {
    while (pNode.next !== null) {
      let t = pNode.next
      if (t.left === pNode)
        return t
      pNode = t
    }
    return null
  }
}
```

## 对称的二叉树

请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

> 思路：节点1的左节点和节点2的右节点判断，节点1的右节点和节点2的左节点判断

```javascript
/* function TreeNode(x) {
    this.val = x;
    this.left = null;
    this.right = null;
} */
function isSymmetrical(pRoot) {
  if (pRoot === null) return true
  return handler(pRoot.left, pRoot.right)
}

function handler(lNode, rNode) {
  if (lNode === null || rNode === null) {
    return lNode === rNode
  }

  return lNode.val === rNode.val &&
         handler(lNode.left, rNode.right) &&
         handler(lNode.right, rNode.left)
}
```

## 按之字形打印二叉树

请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

> 思路：队列 + 记录每层的个数 + flag 标识打印顺序

```javascript
/* function TreeNode(x) {
    this.val = x;
    this.left = null;
    this.right = null;
} */
function Print(pRoot) {
  if (pRoot === null) return []
  let num = 1, queue = [pRoot], flag = true
  let result = []
  while(queue.length > 0) {
    let tNum = 0, tNode = null, tResult = []
    while (num-- > 0) {
      tNode = queue.shift()
      if (flag) tResult.push(tNode.val)
      else tResult.unshift(tNode.val)
      if (tNode.left) {
        queue.push(tNode.left)
        tNum++
      }
      if (tNode.right) {
        queue.push(tNode.right)
        tNum++
      }
    }
    flag = !flag
    result.push(tResult)
    num = tNum
  }
  return result
}
```

## 把二叉树打印成多行

从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

> 思路：队列 + 记录每层的数量

```javascript
/* function TreeNode(x) {
    this.val = x;
    this.left = null;
    this.right = null;
} */
function Print(pRoot) {
  if (pRoot === null) return []
  let num = 1, queue = [pRoot]
  let result = []
  while(queue.length > 0) {
    let tNum = 0, tNode = null, tResult = []
    while (num-- > 0) {
      tNode = queue.shift()
      tResult.push(tNode.val)
      if (tNode.left) {
        queue.push(tNode.left)
        tNum++
      }
      if (tNode.right) {
        queue.push(tNode.right)
        tNum++
      }
    }
    result.push(tResult)
    num = tNum
  }
  return result
}
```

## 二叉搜索树的第k个节点

给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。

> 思路: 二叉搜索树的中序遍历就是从小到大

```javascript
/* function TreeNode(x) {
    this.val = x;
    this.left = null;
    this.right = null;
} */
function KthNode(pRoot, k) {
  if (pRoot === null) return null
  let stack = []
  while (pRoot !== null || stack.length > 0) {
    while(pRoot !== null) {
      stack.push(pRoot)
      pRoot = pRoot.left
    }
    pRoot = stack.pop()
    if (--k === 0) return pRoot
    pRoot = pRoot.right
  }
  return null
}
```

## 滑动窗口的最大值

给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}；

> 思路：记录最大值出现的位置，如果滑动后得到更大的，则更新；否则则判断是否最大值已经滑出，如果已滑出，重新获得最大值

```javascript
"use static"
function maxInWindows(num, size) {
  if (size === 1) return num
  else if (size === 0) return []
  let len = num.length;
  let maxIdx = 0, maxNum = num[0], result = [];
  for (let i = 1; i < len; i++) {
    if (i < size) {
      if (maxNum < num[i]) {
        maxNum = num[i]
        maxIdx = i
      }
      if (i === size - 1) result.push(maxNum)
      continue
    }
    if (num[i] > maxNum) {
      maxNum = num[i]
      maxIdx = i
    } else {
      if (i - size + 1 > maxIdx) {
        maxNum = num[i - size + 1]
        maxIdx = i - size + 1
        for (let j = maxIdx + 1; j <= i; j++) {
          if (maxNum < num[j]) {
            maxNum = num[j]
            maxIdx = j
          }
        }
      }
    }
    result.push(maxNum)
  }
  return result
}
```