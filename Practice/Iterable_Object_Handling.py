import numpy as np

# 序列推导式

# ***************************************************

# 列表推导式 [表达式 for 变量 in 可迭代对象]
squares = [i**2 for i in range(1,11)]  # 普通列表推导式
print(squares)
even_doubled = [i**2 for i in range(1,11) if i % 2 == 0]  # 筛选偶数
scores = [90,60,86,55]  # 多重判断筛选
grades = ["优秀" if x >= 85 else "及格" if x >= 60 else "不及格" for x in scores]
print(even_doubled)
print(grades)
matrix = [[1,2],[3,4],[5,6]]

# 嵌套推导式
flattened = [num for row in matrix for num in row]  # 嵌套循环展平多维数组
print(flattened)
multi_table = [[i*j]for i in range(1,4) for j in range(1,4)]  # 嵌套循环生成多维数组
multi_table2 = [[i*j for i in range(1,4)] for j in range(1,4)]
print(multi_table)
print(multi_table2)

# 字典推导式 {键表达式: 值表达式 for 变量 in 可迭代对象}
squares = {x: x**2 for x in range(1,11)}
print(squares)

# 集合推导式 {表达式 for 变量 in 可迭代对象}  可用于快速去重
words = {"apple","orange","Apple","Orange"}
deduplicated_set = {word.lower() for word in words}
print(deduplicated_set)



# 返回迭代器
# *************************************************************

# 生成器表达式 (表达式 for 变量 in 可迭代对象)
# 惰性计算，就是不会立马计算，调用的时候才会计算，节省内存，用于处理大量数据
gen = (x*2 for x in range(1,10001))  # 在这里返回的是一个生成器，不会立马计算
print(next(gen))
print("分割线")
for i in range(1,5):  # 向下计算了4次，所以生成器也是只能用一次
    print(next(gen))
for num in gen:
    print(num)
    if num == 20:
        break

# zip生成可迭代器（只可以使用一次），list转化生成配对的索引
# zip是短对齐，即两个可迭代对象进行匹配的时候，以较短可迭代对象为基准，长的那个的多余的部分进行舍弃
b = (np.array([1,2,3,4]),np.array([5,6,7,8]))
b1 = list(zip(b[0],b[1]))
print(b1)
b2 = [(int(x),int(y)) for x,y in b1]
print(b2)

# itertools.zip_longest是长对齐，需要先导入模块
from itertools import zip_longest
b3 = [1,2,3,4,5]
b4 = [1,2,3,4]
b5 = zip_longest(b3,b4,fillvalue=0)
b6 = zip(b3,b4)
print(list(b5))
print(list(b6))

# map函数映射
# 对每一个可迭代对象应用函数，返回一个迭代器（类似于生成器）
numbers = np.array([1,2,3])
str_numbers = map(str,numbers)  # 注意这里返回的是迭代器，不是str数组，想要str数组还需要通过list,tuple转化器
print(list(str_numbers))
def square(x):   # 自定义函数进行map映射
    return x**2
print(list(map(square,numbers.tolist())))
cubes = map(lambda x : x**2,numbers.tolist())     # 匿名函数
print(list(cubes))
a = [1,2,3]
b = [4,5,6]
sums = map(lambda x,y: x+y,a,b)  # 多个可迭代对象
print(list(sums))

# 筛选符合条件的元素
numbers = [1,2,3,4,5,6]
even_doubled = filter(lambda x:x%2 == 0,numbers)   # 筛选出来返回迭代器
print(list(even_doubled))

# 同时获取元素和索引，用于需要记录索引的场景
fruits = ["apple","orange","banana"]
for i , fruit in enumerate(fruits):
    if fruit == "orange" :
        print(i)

# 串联多个可迭代对象
from itertools import chain
list1 = [1,2,3,4]
list2 = (5,6,7)
list3 = {8,9,10}
combined = chain(list1,list2,list3)  # 返回一个可迭代对象
print(list(combined))