# 描述: 使用变异系数法 (Coefficient of Variation) 计算客观权重，
#      并从重构后的 preprocessor 模块导入数据预处理函数。

import numpy as np
# 从我们自己创建的 preprocessor.py 文件中，导入两个独立的函数：
# forward_matrix: 仅用于指标正向化
# scale_min_max: 仅用于Min-Max归一化
from pre_processor import forward_matrix, scale_min_max


def calculate_cv_weights(matrix, row_type, row_best):
    """
    执行完整的变异系数法权重计算。

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
    # 对于变异系数法，使用Min-Max归一化可以确保数据非负，避免均值为0，是标准做法。
    print("--- 步骤2: 正在进行Min-Max归一化... ---")
    processed_matrix = scale_min_max(forwarded_matrix)
    print("--- 最终预处理完成的矩阵 ---")
    print(processed_matrix)
    print("-" * 30)

    # 5. 计算变异系数法权重
    # 变异系数法的核心思想是：指标的变异程度越大，越能反映评价对象间的差异，
    # 因此应赋予该指标更高的权重。变异程度用变异系数来衡量。

    # 加上一个极小值 epsilon，防止后续计算中出现除以零的错误
    epsilon = 1e-10

    # 5.1 计算每个指标（列）的均值
    # 均值反映了该指标在所有评价对象上的平均水平
    # axis=0 表示沿着列的方向（即对每个指标）进行计算
    mean_vals = np.mean(processed_matrix, axis=0)

    # 5.2 计算每个指标（列）的标准差
    # 标准差反映了该指标在所有评价对象上的离散程度或波动大小
    std_vals = np.std(processed_matrix, axis=0, ddof=0)

    # 5.3 计算每个指标的变异系数 (V_j = sigma_j / mu_j)
    # 变异系数是一个无量纲的量，它表示标准差相对于均值的比例，
    # 能够客观地反映数据的相对离散程度。
    # 变异系数越大，说明该指标的内部差异越大，区分评价对象的能力越强。
    cv = std_vals / (mean_vals + epsilon)

    # 5.4 归一化处理，得到最终权重
    # W_j = V_j / sum(V)
    # 将每个指标的变异系数除以所有指标变异系数的总和，得到该指标的权重。
    # 这样，变异程度大的指标将分到更多的权重。
    sum_cv = np.sum(cv)
    if sum_cv == 0:
        # 如果所有指标的变异系数都为0（意味着每个指标内部所有评价值都相同），则采用等权重
        n = matrix.shape[1]
        weight = np.full(n, 1/n)
    else:
        weight = cv / sum_cv

    return {
        "mean_values": mean_vals,
        "std_devs": std_vals,
        "cv_values": cv,
        "weights": weight
    }

def display_cv_results(results):
    """
    格式化并打印变异系数法计算的结果。

    参数:
    results (dict): 从 calculate_cv_weights 函数返回的结果字典。
    """
    # 6. 输出最终结果
    print("\n--- 变异系数法计算结果 ---")
    print(f"各指标的均值为: {[f'{m:.4f}' for m in results['mean_values']]}")
    print(f"各指标的标准差为: {[f'{s:.4f}' for s in results['std_devs']]}")
    print(f"各指标的变异系数为: {[f'{c:.4f}' for c in results['cv_values']]}")
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
    cv_results = calculate_cv_weights(matrix_test, row_type_test, row_best_test)

    # 调用显示函数打印结果
    display_cv_results(cv_results)