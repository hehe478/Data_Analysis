import numpy as np

# TOPSIS法，用于计算得分的方法

# 1.设置转化函数
# 所有的参数传入请用np.array类型的数据请用
# 所有的参数都需要注意是都有严重偏离的数据，若有需要修改下列函数防止对结果造成影响
# 即因为若有严重偏离的数据，将会导致其他数据被压缩到一个极小的范围，无法显示差异
# 在数据使用前进行数据预处理
# 1.盖帽法  2.对数转化  3.删除
# 1.1 极大型转化
def to_max(x):  # 其实是毫无用处的一步，只是为了占位，代表这里有一个函数，方便以后如有需要进行修改
    return x
# 1.2 极小型转化
def min_to_max(x):
    max_x = np.max(x)
    x = max_x - x
    return np.array(x)
# 1.3 中间型转化
def mid_to_max(x,best):
    x = np.abs(x - best)
    max_x = np.max(x)
    if max_x == 0:
        return np.ones_like(x)
    x = 1 - x / max_x
    return x
# 1.4 区间型转化
def reg_to_max(x,reg):
    low_x = reg[0]
    high_x = reg[1]  # 取出范围，方便使用
    low = x[x < low_x]
    high = x[x > high_x]
    mid = x[x <= high_x | x >= low_x]  # 对数组进行切片，准备对不同范围进行处理
    max_x = np.max(np.abs(low - low_x),np.abs(high - high_x))  # 求出最大偏离值，进行后续处理
    low = 1 - (low_x - low) / max_x  # 对不同范围的数据进行处理
    high = 1 - (high - high_x) / max_x
    mid = np.ones_like(mid)
    return x
# 2.输入矩阵，设置好评价参数
matrix = np.array([[1,2,3,4],
                   [5,6,7,8],
                   [9,10,11,12],
                   [13,14,15,16]])
row_type = np.array([0,1,2,3])  # 表示每一列的类型，区分数据进行的不同的转化
row_best = [0,0,9,[13,14]] # 表示每一列需要的最佳数值或者区间，即使不需要，为了方便区分也进行了填充
print(row_best)
# 3.处理矩阵，正向化处理
m = matrix.shape  # 获取矩阵的形状
n = m[1]  # 注意评价的类目和被评价的项目分别在行上还是列上，本处为评价的类目在列上
for i in range(n):
    array = matrix[:, i]
    if row_type[i] == 0:
        array = to_max(array)
    elif row_type[i] == 1:
        array = min_to_max(array)
    elif row_type[i] == 2:
        array = mid_to_max(array,row_best[i])
    elif row_type[i] == 3:
        array = reg_to_max(array,row_best[i])
# 4.标准化处理
matrix = matrix.astype(float)
for j in range(n):
    matrix[:,j] = matrix[:,j] / np.sqrt(sum(matrix)**2)
# 5.计算距离，统计得分
# 计算最大最小距离
maxD = np.max(matrix, axis=0)
minD = np.min(matrix, axis=0)
d_z = np.sqrt(np.sum(np.square((matrix - np.tile(maxD, (n, 1)))), axis=1))
d_y = np.sqrt(np.sum(np.square((matrix - np.tile(minD, (n, 1)))), axis=1))

print("每个指标最大值：",maxD)
print("每个指标最小值：",minD)
print("d+向量：",d_z)
print("d-向量：",d_y)

# 计算得分
s = d_y / (d_z + d_y)
Score = 100 * s/sum(s)
for i in range(len(Score)):
    print("第{}个人得分为：{}".format(i+1,Score[i]))