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
print(a1[1:2,2])  # 对数组进行切片
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
d1 = np.eye(3,5)  # 创建一个i=j时候为1其他位置为0的矩阵，可用于创建单位矩阵
print(d1)
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

n = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(n.ravel().shape)  # 将一个数组展平，返回一个新数组
print(n.T.shape)  # 将一个数组转置，返回一个新数组
print(n.reshape(2,6).shape)  # 改变数组的形状，返回一个新数组
print(n.reshape(4,-1).shape)  # -1的含义是指定好其他维度，剩下-1，所在的维度的自动计算
print(n.shape)
print(n.flatten())  # 将原数组展平
n = n.T  # 将原数组进行转置
print(n.shape)
print(n.resize(2,6))  # 在原数组的基础上进行修改形状

# 切分数组
o = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(np.split(o,2,axis = 1))  # 沿着指定的轴，根据参数切分数组，此处是将数组分隔成两个数组
print(np.split(o,(1,2),axis = 0))  # 此处是按照行索引0,1进行切分数组

p = np.array([[1,2,3,4],[5,6,7,8]])
p1 = p.copy()  # 复制一个副本进行操作，方便多次对同一数组进行操作，防止互相污染
p2 = p.view()  # 创建一个视图，方便区分对同一数组进行分阶段操作

# 数组计算操作
q = np.array([1,2,3,4,5])
q1 = np.array([6,7,8,9,10])
print(q + q1)  # 逐元素相加
print(q1 - q)  # 逐元素相减
print(q1 / q)  # 逐元素相除
print(q1 * 1)  # 逐元素相乘
q4 = np.array([[1,2],[3,4]])
q5 = np.array([[5,6],[7,8]])
print(np.dot(q4,q5))  # 适用与下面的函数不一样，建议用于处理向量点积，但也能用于处理二维数组即矩阵
# 例如：np.dot([1,2],[3,4]) 结果：11 ; np.dot([1,2]) 结果：2 ;np.dot(2,[2,3]) 结果：[4,6]
print(q4@q5)  # 也是矩阵乘法，与np.matmul()等价，更适用于高维
print(np.matmul(q4,q5))
# 向量点积，标量相乘用np.dot() 矩阵乘法用@或者np.matmul()
q4_inv = np.linalg.inv(q4)  # 求出逆矩阵
print(q5 @ q4_inv)  # 矩阵中的乘以逆矩阵

print(q.sum())  # 所有元素统一求和
print(q.sum(axis = 0))  # 沿着行进行求和
print(q.prod(axis = 0))  # 累乘
print(q.max(axis = 0))
print(q.min(axis = 0))
print(q.mean(axis = 0))  # 计算平均数
print(q.std(axis = 0))  # 计算标准差
print(np.std(q))  # 各种写法等价，即类方法
# 这些方法都有一些重要的参数
q2 = np.array([[1,2,3,4],[5,6,7,8]])
print(q2.shape)
q3 = np.mean(q2,axis = 0,keepdims = True) # 保留原来的结构，默认是False,导致操作后结构变成了(4,)
print(q3.shape)  # 保留结构，方便广播操作，如不保留后续无法进行广播
# 广播机制详细见广播机制文件BroadcastingMechanism.py

r = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])
eigenvalues , eigenvectors = np.linalg.eig(r)  # 求特征值，特征向量
print("特征值")
print(eigenvalues)  # 特征值
print(eigenvectors)  # 特征向量，注意特征向量应该按列取出来，不能按行取出来
r1 = np.array([[1,2,3],
               [2,1,3],
               [3,3,1]])
print(r @ eigenvectors[:,0]) # 特征向量应该按列取出来，不能按行取出来
print(eigenvalues[0] * eigenvectors[:,0]) # 特征向量应该按列取出来，不能按行取出来
eigenvalues1 , eigenvectors1 = np.linalg.eigh(r1)  # 对称矩阵用这个更加高效
print(eigenvalues1)
print(eigenvectors1)
# 复矩阵用np.linalg.eigvals()  仅仅返回特征值

# 任意矩阵A（不必为方阵）可分解为
# A = UΣV，其中：U和V是正交矩阵，
# Σ是对角矩阵，对角线元素为奇异值。
# 使用np.linalg.svd()求解

s = np.array([[1, 2],
              [3, 4],
              [5, 6]])
U, sigma, VT = np.linalg.svd(s)
print("U矩阵（3x3）：\n", U)
print("奇异值（2个）：", sigma)  # [9.52551809 0.51430058]
print("V^T矩阵（2x2）：\n", VT)

# 方阵的行列式是一个标量，反映矩阵的 “缩放能力”，使用 np.linalg.det()
# 行列式为0的矩阵不可逆
s1 = np.array([[1, 2], [3, 4]])
det_A = np.linalg.det(s1)
print(det_A)

# 方阵主对角线元素之和，使用 np.trace()
# 矩阵的迹等于其特征值之和
s2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
s3 = np.trace(s2)
print(s3)

# 解线性方程组Ax=b对于方阵
# A和向量b，使用 np.linalg.solve()求解
# 求解：
# 1x + 2y = 3
# 4x + 5y = 6
s4 = np.array([[1, 2], [4, 5]])
s5 = np.array([3, 6])
x = np.linalg.solve(s4, s5)
print(x)  # [-1.  2.]（解为x=-1, y=2）
# 验证：A@x 应等于 b
print(np.allclose(s4 @ x, s5))  # True

# 最小二乘法，方程组Ax=b求最小二乘解用 np.linalg.lstsq()：
s6 = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2矩阵（方程数>未知数）
s7 = np.array([7, 8, 9])
x, residuals, rank, s = np.linalg.lstsq(s6, s7, rcond=None)
# rank：整数，表示矩阵的有效秩,反映矩阵中线性无关的行（或列）的最大数量，是判断矩阵 “信息冗余度” 的重要指标
# s : 一维数组，形状为 (min(m, n),)，表示矩阵 A 的奇异值（通过 SVD 分解得到）。
# 奇异值是衡量矩阵 “能量” 的指标，数值越大，对应方向的信息越重要；
# 结合 rcond 可判断哪些奇异值被保留（s > rcond * max(s)），进而理解有效秩的计算依据。
print("最小二乘解：", x)  # 约 [-0.3333  3.6667]
# 残差（拟合误差）
print("残差：", residuals)  # 越小说明拟合越好

# 矩阵的秩是其线性无关的行（或列）的最大数量，用 np.linalg.matrix_rank()
s8 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
rank_s8 = np.linalg.matrix_rank(s8)
print(rank_s8)  # 2（第三行是前两行的线性组合）