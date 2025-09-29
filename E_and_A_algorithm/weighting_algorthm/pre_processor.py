# 文件名: preprocessor.py
# 描述: 一个用于多准则决策分析的数据预处理模块。
# 功能: 提供独立的指标正向化和多种标准化/归一化方法，可供用户自由组合调用。

import numpy as np


# ==============================================================================
# 函数一：指标正向化
# ==============================================================================
def forward_matrix(matrix, row_type, row_best):
    """
    仅对评价矩阵进行“正向化”处理，不进行任何标准化或归一化。
    正向化是确保所有指标的趋势一致（都变成越大越好）。
    对于已经是极大型（效益型）的指标，此函数会保留其原始值。

    参数:
    - matrix (np.ndarray): 原始评价矩阵 (m个对象, n个指标)。
    - row_type (np.ndarray): 长度为n的一维数组，定义每个指标的类型。
                               0: 极大型, 1: 极小型, 2: 中间型, 3: 区间型。
    - row_best (list): 长度为n的列表，为中间型或区间型指标提供最优值/区间。

    返回:
    - np.ndarray: 所有指标都已转换为“越大越好”趋势的矩阵。
    """

    # --- 内部辅助函数，用于转化各类指标 ---

    # 1.1 极小型转化为极大型（成本型 -> 效益型）
    # 核心思想：原始值越小，转化后的值越大。
    # 使用 (max - x) / (max - min) 公式，可以将结果归一化到 [0, 1] 区间。
    # 原始值最小的x，其结果为 (max-min)/(max-min)=1；原始值最大的x，其结果为0。
    def _min_to_max(x):
        max_x = np.max(x)  # 找到该指标（列）的最大值
        min_x = np.min(x)  # 找到该指标（列）的最小值
        if max_x == min_x: return np.ones_like(x)  # 特殊情况：如果所有值都相等，则无法区分优劣，全部赋值为1
        return (max_x - x) / (max_x - min_x)

    # 1.2 中间型转化为极大型
    # 核心思想：越接近最佳值 `best` 的数据，转化后的值越大。
    # 计算每个值与 `best` 的绝对差距，然后用 1 减去这个差距的相对大小。
    def _mid_to_max(x, best):
        m1 = np.abs(x - best)  # 计算每个元素与最佳值的绝对距离
        max_m = np.max(m)  # 找到这些距离中的最大值
        if max_m == 0: return np.ones_like(x)  # 特殊情况：如果所有值都等于best，说明它们都是最优的，全部赋值为1
        return 1 - m1 / max_m  # 距离越大，1-m/max_m 的值越小，实现了正向化

    # 1.3 区间型转化为极大型
    # 核心思想：如果值落 在指定区间 `reg` 内则最优，落在区间外的值离区间越远越差。
    def _reg_to_max(x, reg):
        low_x, high_x = reg  # 提取区间的下限和上限
        # 使用布尔索引创建三个掩码（mask），用于标识数据的位置
        mask_low = x < low_x  # 标记所有小于区间 下限的值
        mask_high = x > high_x  # 标记所有大于区间 上限的值
        mask_mid = (~mask_low) & (~mask_high)  # 标记所有在区间内部（包括边界）的值

        # 计算所有偏离区间的值与最近边界的距离
        deviations = np.concatenate([low_x - x[mask_low], x[mask_high] - high_x])
        if len(deviations) == 0: return np.ones_like(x)  # 如果没有偏离的值，说明所有数据都在最优区间内，全部赋值为1
        max_deviation = np.max(deviations)  # 找到最大的偏离距离

        # 初始化一个与输入x形状和类型都相同的全零数组，用于存放处理后的结果
        x_processed = np.zeros_like(x, dtype=float)
        # 根据掩码对不同位置的数据进行赋值
        x_processed[mask_low] = 1 - (low_x - x[mask_low]) / max_deviation  # 低于下限的值：离得越远，值越小
        x_processed[mask_high] = 1 - (x[mask_high] - high_x) / max_deviation  # 高于上限的值：离得越远，值越小
        x_processed[mask_mid] = 1  # 区间内的值，都是最优，赋值为1
        return x_processed

    # --- 主循环：遍历每一列并应用相应的正向化函数 ---
    m, n = matrix.shape  # 获取原始矩阵的行数m（评价对象数）和列数n（指标数）
    # 创建一个和原始矩阵一样大小的全零浮点数矩阵，用于存储正向化后的结果
    forwarded_matrix = np.zeros_like(matrix, dtype=float)

    for i in range(n):  # 遍历每一列（即每一个指标）
        array = matrix[:, i]  # 取出当前循环处理的第 i 列数据
        if row_type[i] == 0:  # 判断指标类型是否为 "极大型"
            forwarded_array = array  # 如果是，则无需处理，直接使用原数据
        elif row_type[i] == 1:  # 判断指标类型是否为 "极小型"
            forwarded_array = _min_to_max(array)  # 调用极小型转化函数
        elif row_type[i] == 2:  # 判断指标类型是否为 "中间型"
            forwarded_array = _mid_to_max(array, row_best[i])  # 调用中间型转化函数，并传入对应的最佳值
        elif row_type[i] == 3:  # 判断指标类型是否为 "区间型"
            forwarded_array = _reg_to_max(array, row_best[i])  # 调用区间型转化函数，并传入对应的最佳区间
        else:
            forwarded_array = array  # 如果 row_type 中的类型未定义，则默认不处理

        # 将处理完的这一列数据，赋值给新矩阵的对应列
        forwarded_matrix[:, i] = forwarded_array

    return forwarded_matrix  # 返回全部指标都已正向化的矩阵


# ==============================================================================
# 函数组二：标准化与归一化方法
# 这些函数假设输入的矩阵已经是正向化的（即所有指标都是越大越好）。
# ==============================================================================

def scale_min_max(matrix):
    """
    对矩阵的每一列进行 Min-Max 归一化。
    将所有数据线性映射到 [0, 1] 区间。这是最常用的 归一化方法之一。

    参数:
    - matrix (np.ndarray): 一个已经正向化的矩阵。

    返回:
    - np.ndarray: Min-Max 归一化后的矩阵。
    """
    n = matrix.shape[1]  # 获取矩阵的列数
    # 创建一个与输入矩阵同样大小的全零浮点型矩阵，用于存储归一化结果
    scaled_matrix = np.zeros_like(matrix, dtype=float)

    # 逐列进行处理
    for i in range(n):  # 遍历每一列
        col = matrix[:, i]  # 获取当前列的数据
        min_val = np.min(col)  # 计算该列的最小值
        max_val = np.max(col)  # 计算该列的最大值

        # 处理特殊情况：如果一列中的所有值都相同，那么 max_val 和 min_val 会相等
        if max_val == min_val:
            # 此时分母为0，无法进行计算。通常将这一列的所有值都设为1。
            scaled_matrix[:, i] = np.ones_like(col)
        else:
            # 应用 Min-Max 归一化公式: (x - min) / (max - min)
            scaled_matrix[:, i] = (col - min_val) / (max_val - min_val)

    return scaled_matrix  # 返回归一化后的矩阵


def scale_z_score(matrix, ensure_positive=False):
    """
    对矩阵的每一列进行 Z-score 标准化。
    将所有数据转换为均值为0，标准差为1的分布。它对异常值不敏感。

    参数:
    - matrix (np.ndarray): 一个已经正向化的矩阵。
    - ensure_positive (bool): 是否要确保所有输出值为正数（通过线性平移）。
                              某些算法（如熵权法）要求输入为正，此时需要将此参数设为True。

    返回:
    - np.ndarray: Z-score 标准化后的矩阵。
    """
    # 计算每一列的均值（mean）和标准差（std）。axis=0表示沿列（纵向）计算。
    mean_vals = np.mean(matrix, axis=0)
    std_vals = np.std(matrix, axis=0)

    # 定义一个极小值 epsilon，防止标准差为0时出现除零错误。
    epsilon = 1e-10
    # 应用 Z-score 标准化公式: (x - mean) / std
    standardized_matrix = (matrix - mean_vals) / (std_vals + epsilon)

    # 如果 `ensure_positive` 参数为 True，则执行以下代码块
    if ensure_positive:
        # 找到整个标准化后矩阵中的最小值
        min_val = np.min(standardized_matrix)
        # 如果最小值小于0，说明矩阵中存在负数
        if min_val < 0:
            # 将矩阵中的所有元素都减去这个最小值（即加上最小值的绝对值），
            # 这样可以使新的最小值为0，从而保证所有数据都是非负的。
            standardized_matrix = standardized_matrix - min_val

    return standardized_matrix  # 返回标准化后的矩阵


def scale_vector(matrix):
    """
    对矩阵的每一列进行向量归一化（Vector Normalization）。
    这种方法常用于TOPSIS分析中，其目的是使每一列（指标）的欧几里得范数（L2范数）等于1。

    参数:
    - matrix (np.ndarray): 一个已经正向化的矩阵。

    返回:
    - np.ndarray: 向量归一化后的矩阵。
    """
    n = matrix.shape[1]  # 获取矩阵的列数
    # 创建一个与输入矩阵同样大小的全零浮点型矩阵，用于存储归一化结果
    scaled_matrix = np.zeros_like(matrix, dtype=float)

    for i in range(n):  # 遍历每一列
        col = matrix[:, i]  # 获取当前列的数据
        # 计算该列的L2范数，即向量的欧几里得长度：sqrt(x1^2 + x2^2 + ... + xn^2)
        l2_norm = np.sqrt(np.sum(col ** 2))

        # 处理特殊情况：如果L2范数为0，意味着该列所有元素都是0
        if l2_norm == 0:
            scaled_matrix[:, i] = col  # 此时无法进行除法运算，保持原样（全为0）
        else:
            # 将该列的每个元素都除以其L2范数
            scaled_matrix[:, i] = col / l2_norm

    return scaled_matrix  # 返回归一化后的矩阵