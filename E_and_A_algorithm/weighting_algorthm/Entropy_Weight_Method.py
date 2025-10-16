# 描述: 使用熵权法 (Entropy Weight Method) 计算客观权重，
#      并从重构后的 preprocessor 模块导入数据预处理函数。

import numpy as np


# 从我们自己创建的 preprocessor.py 文件中，导入两个独立的函数：
# forward_matrix: 仅用于指标正向化
# scale_min_max: 仅用于Min-Max归一化
from pre_processor import forward_matrix, scale_min_max



def calculate_entropy_weights(matrix, row_type, row_best):
    """
    执行完整的熵权法权重计算。

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
    # 熵权法需要非负输入，Min-Max归一化到[0, 1]区间是标准且安全的做法。
    print("--- 步骤2: 正在进行Min-Max归一化... ---")
    processed_matrix = scale_min_max(forwarded_matrix)
    print("--- 最终预处理完成的矩阵 ---")
    print(processed_matrix)
    print("-" * 30)

    # 5. 计算熵权法权重
    # 熵权法的核心思想是：指标的信息熵越小，说明其信息的无序程度越低，
    # 提供的信息量越多，因此在评价体系中应赋予更高的权重。

    # 加上一个极小值 epsilon，防止后续计算中出现 log(0) 的数学错误
    epsilon = 1e-10

    # 5.1 计算每个评价对象在每个指标下的比重 P_ij
    # P_ij = x_ij / sum(x_i) for i=1 to m
    # P矩阵的每一列之和为1
    # 首先计算每一列（指标）的总和
    col_sums = np.sum(processed_matrix, axis=0)
    # 然后计算比重矩阵 P。为防止分母为0，对列和为0的情况进行处理
    P = processed_matrix / (col_sums + epsilon)

    # 5.2 计算每个指标的信息熵 e_j
    # e_j = -k * sum(P_ij * ln(P_ij)) for i=1 to m, 其中 k = 1 / ln(m)
    # m 是评价对象的数量
    m = matrix.shape[0]
    k = -1 / np.log(m)

    # 熵值e是一个包含n个元素的一维向量，每个元素对应一个指标的熵值
    # 熵值越大，表示该指标下的数据越趋于一致，信息量越小
    e = k * np.sum(P * np.log(P + epsilon), axis=0)

    # 5.3 计算信息冗余度 (或称差异性系数) d_j
    # d_j = 1 - e_j
    # 信息冗余度与信息熵相反，d_j 越大，说明该指标提供的信息越多，越重要
    d = 1 - e

    # 5.4 归一化处理，得到最终权重
    # W_j = d_j / sum(d)
    # 将每个指标的信息冗余度除以总冗余度，得到最终的权重
    # 检查冗余度总和是否为0
    sum_d = np.sum(d)
    if sum_d == 0:
        # 如果所有指标冗余度都为0（例如每个指标内部值都相同），则所有指标同等重要
        n = matrix.shape[1]
        weight = np.full(n, 1 / n)
    else:
        weight = d / sum_d

    return {
        "entropy": e,
        "redundancy": d,
        "weights": weight
    }


def display_entropy_results(results):
    """
    格式化并打印熵权法计算的结果。

    参数:
    results (dict): 从 calculate_entropy_weights 函数返回的结果字典。
    """
    # 6. 输出最终结果
    print("\n--- 熵权法计算结果 ---")
    print(f"各指标的信息熵(e): {[f'{val:.4f}' for val in results['entropy']]}")
    print(f"各指标的信息冗余度(d): {[f'{val:.4f}' for val in results['redundancy']]}")
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
    entropy_results = calculate_entropy_weights(matrix_test, row_type_test, row_best_test)

    # 调用显示函数打印结果
    display_entropy_results(entropy_results)