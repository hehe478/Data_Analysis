import numpy as np

# 熵权法(Entropy Weight Method)，用于计算权重的方法


# ******************************************************************************（可以修改的地方）


# 1.设置转化函数
# 所有的参数传入请用np.array类型的数据请用
# 将数据进行正向化以及标准化以消除量纲的影响
# 此处的标准化为最大最小范围内的放缩，如需要更严格的标准化，建议使用Z_Score标准化
# Z_SCORE标准化后要保证数据全是正值，如有负值，建议加常数偏移
# 所有的参数都需要注意是都有严重偏离的数据，若有需要修改下列函数防止对结果造成影响
# 即因为若有严重偏离的数据，将会导致其他数据被压缩到一个极小的范围，无法显示差异
# 在数据使用前进行数据预处理
# 1.盖帽法  2.对数转化  3.删除
# 1.1 极大型转化
def to_max(x):  # 对数据进行标准化处理，压缩到0-1之间，保留信息的同时降低量纲差异
    x_max = np.max(x)
    x_min = np.min(x)
    return (x - x_min)/(x_max - x_min)
# 1.2 极小型转化
def min_to_max(x):
    max_x = np.max(x)
    min_x = np.min(x)
    return (max_x - x)/(max_x - min_x)
# 1.3 中间型转化
def mid_to_max(x, best):
    x = np.abs(x - best)
    max_x = np.max(x)
    if max_x == 0:
        return np.ones_like(x)  # 如果所有值都等于best，则都是最优，返回1
    return 1 - x / max_x
# 1.4 区间型转化
def reg_to_max(x, reg):
    low_x, high_x = reg[0], reg[1]  # 取出范围，方便使用
    # 使用&进行元素级逻辑与运算，并用括号明确优先级
    mask_low = x < low_x
    mask_high = x > high_x
    mask_mid = ~mask_low & ~mask_high  # 既不小于下限也不大于上限
    # 计算最大偏离值
    deviations = np.concatenate([low_x - x[mask_low], x[mask_high] - high_x])
    if len(deviations) == 0:  # 所有值都在区间内
        return np.ones_like(x)
    max_deviation = np.max(deviations)
    # 对不同范围的数据进行处理
    x_processed = np.zeros_like(x, dtype=float)
    x_processed[mask_low] = 1 - (low_x - x[mask_low]) / max_deviation
    x_processed[mask_high] = 1 - (x[mask_high] - high_x) / max_deviation
    x_processed[mask_mid] = 1  # 区间内的值为1
    return x_processed

# 2.输入评价矩阵
matrix = np.array([[1,2,3,4],
              [8,7,6,5],
              [9,10,11,12],
              [13,14,15,16]])
row_type = np.array([0, 1, 2, 3])  # 表示每一列的类型，区分数据进行的不同的转化
row_best = [0, 0, 9, [13, 14]]  # 表示每一列需要的最佳数值或者区间，即使不需要，为了方便区分也进行了填充


# ******************************************************************************（其他地方可复用）


# 3.对矩阵进行处理
m, n = matrix.shape  # 获取矩阵的形状
processed_matrix = matrix.copy().astype(float)

for i in range(n):
    array = matrix[:, i]
    if row_type[i] == 0:
        processed_array = to_max(array)
    elif row_type[i] == 1:
        processed_array = min_to_max(array)
    elif row_type[i] == 2:
        processed_array = mid_to_max(array, row_best[i])
    elif row_type[i] == 3:
        processed_array = reg_to_max(array, row_best[i])
    processed_matrix[:, i] = processed_array
# 4. 计算信息熵和权重
# 加上一个极小值epsilon，防止log(0)的错误
epsilon = 1e-10

# 计算比重 Pij，注意要使用 processed_matrix 并且按列(axis=0)求和
P = processed_matrix / np.sum(processed_matrix, axis=0)

# 计算每个指标的信息熵 e，结果是一个向量
# 使用 P * np.log(P + epsilon) 来计算，np.sum按列(axis=0)求和
k = -1 / np.log(m)
e = k * np.sum(P * np.log(P + epsilon), axis=0)

# 计算信息冗余度 d (差异性系数)
d = 1 - e

# 5. 计算最终权重
# <<< 修正点: d.sum() 是对所有指标的差异性系数求和
weight = d / np.sum(d)

formatted_weights = [f"{w:.4f}" for w in weight]  # 规范格式输出
print(f"各指标的权重为: {formatted_weights}")
print("\n权重加和:", np.sum(weight))