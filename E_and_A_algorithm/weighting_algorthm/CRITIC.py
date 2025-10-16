# 描述: 使用 CRITIC 法计算客观权重，并从重构后的 preprocessor 模块导入数据预处理函数。

import numpy as np


# 从我们自己创建的 preprocessor.py 文件中，导入两个独立的函数：
# forward_matrix: 仅用于指标正向化
# scale_min_max: 仅用于Min-Max归一化
from pre_processor import forward_matrix, scale_min_max


def calculate_critic_weights(matrix, row_type, row_best):
    """
    执行完整的CRITIC法权重计算，包括数据预处理和信息量计算。

    参数:
    matrix (np.ndarray): 原始评价矩阵。
    row_type (np.ndarray): 每个指标的类型定义数组。
    row_best (list): 为中间型或区间型指标提供的最优值/区间。

    返回:
    dict: 一个包含所有计算结果的字典。
    """
    # 4. 数据预处理 (采用新的两步法)
    # 这种分步调用的方式，让数据处理的每一步意图都非常明确。

    # 4.1 第一步：指标正向化
    # 调用 forward_matrix 函数，确保所有指标都变为“越大越好”的趋势。
    print("--- 步骤1: 正在进行指标正向化... ---")
    forwarded_matrix = forward_matrix(matrix, row_type, row_best)
    print("--- 正向化后的矩阵 ---")
    print(forwarded_matrix)
    print("-" * 30)

    # 4.2 第二步：Min-Max 归一化
    # 调用 scale_min_max 函数，对已经正向化的矩阵进行归一化，消除量纲影响。
    # 对于CRITIC法，使用Min-Max是标准做法，以保留原始数据的波动信息。
    print("--- 步骤2: 正在进行Min-Max归一化... ---")
    processed_matrix = scale_min_max(forwarded_matrix)
    print("--- 最终预处理完成的矩阵 ---")
    print(processed_matrix)
    print("-" * 30)

    # 5. 计算CRITIC权重
    # CRITIC法通过指标的“对比强度”和“冲突性”来综合衡量其重要性

    # 5.1 计算指标的变异性（对比强度），用标准差表示
    # 标准差越大，代表该指标在不同评价对象间的数值差异越大，区分能力越强，包含的信息越多
    std_devs = np.std(processed_matrix, axis=0, ddof=0)

    # 5.2 计算指标之间的冲突性，用相关系数表示
    # 首先，计算归一化后矩阵各列（指标）之间的皮尔逊相关系数矩阵
    # row_var=False 表示每一列代表一个变量（指标），每一行代表一个观测值
    corr_matrix = np.corrcoef(processed_matrix, rowvar=False)

    # 然后，计算每个指标的冲突性。冲突性 = Σ(1 - r_ij)
    # r_ij 是指标 i 和 j 之间的相关系数。如果两个指标相关性强（r接近1），则(1-r)小，信息重叠度高，冲突性小。
    # 反之，如果两个指标不相关（r接近0），则(1-r)大，信息重叠度低，该指标的独立性强。
    conflict = np.sum(1 - corr_matrix, axis=1)

    # 5.3 计算每个指标的信息承载量 C_j
    # C_j = std_dev_j * conflict_j
    # 信息量综合了指标自身的变异性（对比强度）和与其他指标的冲突性（独立性）。
    # C值越大，说明该指标不仅内部差异大，而且与其他指标的关联度低，能提供更多独特信息，因此越重要。
    info_amount = std_devs * conflict

    # 5.4 归一化处理，得到最终权重
    # W_j = C_j / sum(C)
    # 检查信息量总和是否为0，避免除零错误
    sum_info = np.sum(info_amount)
    if sum_info == 0:
        # 如果所有指标信息量都为0（例如每个指标内部值都相同），则所有指标同等重要
        n = matrix.shape[1]
        weight = np.full(n, 1 / n)
    else:
        weight = info_amount / sum_info

    return {
        "std_devs": std_devs,
        "conflict": conflict,
        "info_amount": info_amount,
        "weights": weight
    }


def display_critic_results(results):
    """
    格式化并打印CRITIC法计算的结果。

    参数:
    results (dict): 从 calculate_critic_weights 函数返回的结果字典。
    """
    # 6. 输出最终结果
    print("\n--- CRITIC 法计算结果 ---")
    print(f"各指标的标准差 (对比强度): {[f'{s:.4f}' for s in results['std_devs']]}")
    print(f"各指标的冲突性: {[f'{c:.4f}' for c in results['conflict']]}")
    print(f"各指标的信息承载量: {[f'{i:.4f}' for i in results['info_amount']]}")
    print("-" * 30)

    formatted_weights = [f"{w:.4f}" for w in results['weights']]
    print(f"各指标的权重为: {formatted_weights}")
    print("\n权重加和:", np.sum(results['weights']))


# 当这个脚本作为主程序运行时，执行以下代码
if __name__ == '__main__':
    # ******************************************************************************（可以修改的地方）

    # 1.输入原始评价矩阵
    # 假设有 m 个评价对象, n 个评价指标
    # matrix[i, j] 表示第 i 个对象在第 j 个指标下的值
    matrix_test = np.array([[1, 2, 3, 4],
                            [8, 7, 6, 5],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16]], dtype=float)

    # 2.定义每个指标的类型
    # 0: 极大型, 1: 极小型, 2: 中间型, 3: 区间型
    row_type_test = np.array([0, 1, 2, 3])

    # 3.为中间型或区间型指标提供最优值/最优区间
    # 对于非中间/区间型指标，可以用任意值（如0）填充占位
    row_best_test = [0, 0, 9, [13, 14]]

    # ******************************************************************************（其他地方可复用）

    # 调用核心函数执行计算
    critic_results = calculate_critic_weights(matrix_test, row_type_test, row_best_test)

    # 调用显示函数打印结果
    display_critic_results(critic_results)