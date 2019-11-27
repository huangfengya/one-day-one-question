## 数组扁平化

1. 递归

> 思路: 主要是使用 apply 会对数组进行展开的思路递归

```javascript
function flatten(a) {
  let t = []
  for (let val of a) {
    if (Array.isArray(val)) {
      t = Array.prototype.concat.apply(t, flatten(val))
    } else {
      t.push(val)
    }
  }
  return t
}
```

2. reduce

> 思路：reduce 接受两个参数，回调和初始值，实际上也是递归

```javascript
function flatten(arr) {
  return arr.reduce((a, b) =>
    [...a, ...(Array.isArray(b) ? flatten(b) : [b])]
  , [])
}
```

3. 扩展运算符 + concat

> 思路：concat 合并数组时，如果该参数不是数组，那么直接插入，如果是数组，会选择数组中的参数

```javascript
function flatten(arr) {
  while(arr.some(val => Array.isArray(val))) {
    arr = [].concat(...arr)
  }
  return arr
}
```

## 打乱数组

> 思路：洗牌算法：每次从数组里面取出一个值(不放回去)，直到数组为空，而洗牌算法正是如此，从 (i, len - 1) 的索引中进行交换值，对于已经交换过的，不再参与交换

```javascript
class Solution {
    /**
     * @param {number[]} nums
     */
    constructor(nums) {
        this.nums = nums
    }

    /**
     * Returns a random shuffling of the array.
     * @return {number[]}
     */
    shuffle() {
        let t = [...this.nums]
        let len = this.nums.length
        for (let i = 0; i < len; i++) {
            let idx = (Math.random() * (len - i) | 0) + i
            if (idx !== i) [t[i], t[idx]] = [t[idx], t[i]]
        }
        return t
    }
    /**
     * Resets the array to its original configuration and return it.
     * @return {number[]}
     */
    reset() {
        return this.nums
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * var obj = new Solution(nums)
 * var param_1 = obj.reset()
 * var param_2 = obj.shuffle()
 */
```
