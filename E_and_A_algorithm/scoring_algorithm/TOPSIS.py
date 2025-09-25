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
    return max_x - x
# 1.3 中间型转化
def mid_to_max(x, best):
    x = np.abs(x - best)
    max_x = np.max(x)
    if max_x == 0:
        return np.ones_like(x)
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


# 2.输入矩阵，设置好评价参数
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])
row_type = np.array([0, 1, 2, 3])  # 表示每一列的类型，区分数据进行的不同的转化
row_best = [0, 0, 9, [13, 14]]  # 表示每一列需要的最佳数值或者区间，即使不需要，为了方便区分也进行了填充


# 3.处理矩阵，正向化处理
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

# 4.标准化处理
for j in range(n):
    col_sum_sq = np.sum(processed_matrix[:, j] ** 2)
    if col_sum_sq != 0:
        processed_matrix[:, j] = processed_matrix[:, j] / np.sqrt(col_sum_sq)
# 如果全部都是0才能使col_sum_sq = 0,那个时候就不需要转化了


# 5.计算距离，统计得分
# 计算理想解和负理想解
ideal_best = np.max(processed_matrix, axis=0)  # 统计所有的研究对象中最优和最差，需要按行进行处理
ideal_worst = np.min(processed_matrix, axis=0)

# 计算到理想解和负理想解的距离
d_best = np.sqrt(np.sum(np.square(processed_matrix - ideal_best), axis=1))  # 将每一个研究对象的各个项目和最优最差的数据进行对比，按列进行处理
d_worst = np.sqrt(np.sum(np.square(processed_matrix - ideal_worst), axis=1))

print("每个指标理想最佳值：", ideal_best)
print("每个指标理想最差值：", ideal_worst)
print("到理想最佳解的距离：", d_best)
print("到理想最差解的距离：", d_worst)

# 计算得分
scores = d_worst / (d_best + d_worst)
normalized_scores = 100 * scores / np.sum(scores)  # 转化为百分制

for i in range(len(normalized_scores)):
    print(f"第{i + 1}个评价对象得分为：{normalized_scores[i]:.2f}")