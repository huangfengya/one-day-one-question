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