import numpy as np
# 从我们自己创建的 preprocessor.py 文件中，导入两个独立的函数：
# forward_matrix: 仅用于指标正向化
# scale_vector: 仅用于TOPSIS专属的向量归一化
from E_and_A_algorithm.weighting_algorthm.pre_processor import forward_matrix,scale_vector
# 1.输入原始评价矩阵，并设置好评价参数
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]], dtype=float)

# 2.定义每个指标的类型
# 0: 极大型, 1: 极小型, 2: 中间型, 3: 区间型
row_type = np.array([0, 1, 2, 3])

# 3.为中间型或区间型指标提供最优值/最优区间
# 对于非中间/区间型指标，可以用任意值（如0）填充占位
row_best = [0, 0, 9, [13, 14]]

# ******************************************************************************（其他地方可复用）

# 4. 数据预处理 (采用新的两步法)
# 通过分步调用，数据处理的流程和意图一目了然。

# 4.1 第一步：指标正向化
# 调用 forward_matrix 函数，确保所有指标都变为“越大越好”的趋势。
print("--- 步骤1: 正在进行指标正向化... ---")
forwarded_matrix = forward_matrix(matrix, row_type, row_best)
print("--- 正向化后的矩阵 ---")
print(forwarded_matrix)
print("-" * 30)

# 4.2 第二步：向量归一化
# 调用 scale_vector 函数，这是TOPSIS方法的标准归一化步骤，
# 用于消除量纲影响，并为后续的距离计算做准备。
print("--- 步骤2: 正在进行向量归一化... ---")
processed_matrix = scale_vector(forwarded_matrix)
print("--- 最终预处理完成的矩阵 ---")
print(processed_matrix)
print("-" * 30)


# 5.计算距离与得分
# 5.1 确定最优理想解 (Ideal Best) 和最劣理想解 (Ideal Worst)
# 最优理想解由处理后矩阵的每列最大值构成
ideal_best = np.max(processed_matrix, axis=0)
# 最劣理想解由处理后矩阵的每列最小值构成
ideal_worst = np.min(processed_matrix, axis=0)

# 5.2 计算各评价对象到最优和最劣理想解的欧几里得距离
# axis=1 表示按行计算，即为每个评价对象计算一个距离值
d_best = np.sqrt(np.sum(np.square(processed_matrix - ideal_best), axis=1))
d_worst = np.sqrt(np.sum(np.square(processed_matrix - ideal_worst), axis=1))

# 5.3 计算各评价对象的最终得分
# 得分公式 C_i = d_worst / (d_best + d_worst)
# 该得分是一个介于0和1之间的值，越接近1表示结果越优
scores = d_worst / (d_best + d_worst)


# 6. 输出最终结果
print("\n--- TOPSIS 法计算结果 ---")
print(f"每个指标的最优理想值 (Ideal Best): {[f'{val:.4f}' for val in ideal_best]}")
print(f"每个指标的最劣理想值 (Ideal Worst): {[f'{val:.4f}' for val in ideal_worst]}")
print(f"各评价对象到最优解的距离 (D+): {[f'{val:.4f}' for val in d_best]}")
print(f"各评价对象到最劣解的距离 (D-): {[f'{val:.4f}' for val in d_worst]}")
print("-" * 30)

print("--- 最终得分 ---")
for i, score in enumerate(scores):
    print(f"第 {i + 1} 个评价对象的得分为 (0-1范围): {score:.4f}")

# 您也可以像之前一样，将得分转化为百分制进行输出
# normalized_scores = 100 * scores / np.sum(scores)
# print("\n--- 百分制得分 ---")
# for i, score in enumerate(normalized_scores):
#     print(f"第 {i + 1} 个评价对象的百分制得分为: {score:.2f}")