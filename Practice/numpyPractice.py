import numpy as np
a = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
print(a)
print(a[2])  # 索引访问
print(a[1:3])  # 这切片是1到3-1
print(a[a > 3]) # 条件进行索引
five_up = (a > 5) | (a == 5) # 表示真值数组，也可以用np.logical_or(a > 5,a == 5)代替
print(five_up)
print(np.where(a > 5))  # 用于寻找符合条件的元素索引
a1 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
six_up = (a1 > 6) | (a1 == 6)
a2 = np.where(six_up)
print(list(zip(a2[0],a2[1])))  # 这里的zip是为了将这行列索引互相配对组合，原本是纯行索引和纯列索引
#  输出的是np.int64类型的数，但是不影响数值计算，如果想要进行转化，1.转化为列表再zip   2.zip后再转化
print(list(zip(a2[0].tolist(),a2[1].tolist())))  #  转化成列表在进行zip
a3 = list(zip(a2[0],a2[1]))  #  zip后在进行转化
a4 = [(int(x),int(y)) for x,y in a3]  #  列表推导式，详情见可迭代对象处理Iterable_Object_Handling.py
print(a4)

# a[start:stop:step] start开始索引 stop结束索引 step步长
# start没写默认是0,stop没写默认是结束,step没写默认是1
print(a[0:6:2]) # 从0到6-1，每隔2个取出，可用于间隔采样，奇偶采样
print(a[::-1]) # 反转数组
print(a[:3])

# 切片返回的是视图，不是副本，所以对切片里的值进行更改会导致原数组产生变化

b = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(b)
print(a.ndim)   # 数组维度是1
print(b.ndim)   # 数组维度是2
print(a.shape)   # 数组每个维度上的元素个数
print(b.shape)
[b1,b2] = b.shape  # 提取数组每个维度上的元素个数
print(b1,b2)
print(b.size)   #  数组总元素数
print(b.dtype)   #  数组的元素类型


c = np.zeros((2,2))  #  全是0的数组
print(c)
d = np.ones((3,3))   #  全是1的数组
print(d)
e = np.empty((2,2))   #  生成一个元素是随机的数组，目的是追求生成速度
print(e)
f = np.arange(2,9,2)  #  start,stop,step
print(f)
g = np.arange(4)   #  生成0到4-1的数组
print(g)
h = np.linspace(0,10,5,axis = 0,dtype = np.int64)  #  axis指定轴，dtype指定类型
print(h)

# 排序
i = np.array([2,4,6,3,9,1,5])
j = np.sort(i)   #  返回排序后的副本，原来的不变
print(i,j)
i.sort()   #  对原来的数组进行排序
print(i)

# 对数组进行拼接
k = np.array([[1,3,5,7],[2,4,6,8]])
l = np.array([[2,4,6,8],[1,3,5,7]])
print(np.concatenate((k,l),axis = 1)) # 注意要加一个括号让这两个形成元组
print(k)
print(l)

m = np.array([1,2,3,4,5])
print(m.shape)
m1 = m[:,np.newaxis]  # 增加一个新的轴，维度
m2 = m[np.newaxis,:]
m3 = np.expand_dims(m,axis = 0)
print(m1.shape,m2.shape,m3.shape)
print(m1,m2,m3)

