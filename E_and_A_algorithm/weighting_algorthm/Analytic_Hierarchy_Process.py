import numpy as np

# 层次分析法(Analytic Hierarchy Process)，用于计算权重
# 本代码可复用，如需修改，仅需修改判断矩阵部分 即第1部分
RI = np.array([0, 0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45])  # 导入RI，后面用于一致性检验
# 特地多打了一个0，为的是能够和数组维度精准匹配

def calculate_ahp_weights(matrix):
    """
    根据输入的判断矩阵，执行AHP计算，包括验证、权重计算和一致性检验。

    参数:
    A (np.ndarray): 用户定义的判断矩阵。

    返回:
    dict: 包含计算结果的字典，包括权重、一致性比例和检验是否通过。

    异常:
    ValueError: 如果矩阵不满足正互反矩阵的条件。
    """
    # 2.计算特征值和特征向量，找到最大特征值与其所对应的特征向量
    n, m = matrix.shape  # 获取矩阵的维度，只是取了两个不同变量名，实际上m,n数值是一致的
    # 验证是否为正互反矩阵
    # 因为若是输入的矩阵过大或者复杂可能存在一定错误，所以进行一定判断保证输入矩阵的正确性
    for i in range(n):
        if not np.isclose(matrix[i, i], 1):
            raise ValueError(f"判断矩阵对角线元素必须为1，第{i + 1}行第{i + 1}列错误")
        for j in range(i + 1, n):
            if not np.isclose(matrix[i, j], 1 / matrix[j, i]):
                raise ValueError(f"判断矩阵不满足互反性，第{i + 1}行第{j + 1}列与第{j + 1}行第{i + 1}列错误")
        if np.any(matrix[i, :] <= 0):
            raise ValueError(f"判断矩阵元素必须为正数，第{i + 1}行存在非正数")

    # 开始获取特征值与特征向量
    eigenvalues, eigenvectors = np.linalg.eig(matrix)  # 计算特征值和特征向量
    maxEigValIndex = np.argmax(eigenvalues.real)  # 最大特征值对应索引 (取实部 保证准确性)
    maxEigVal = eigenvalues[maxEigValIndex]  # 最大特征值
    maxEigVec = eigenvectors[:, maxEigValIndex]  # 最大特征值所对应的特征向量,注意此处是列，不是行，应该取列

    # 3.一致性检验
    # CI = (maxEigVal - n）/（n - 1），最大特征值和矩阵的维度
    # CR = CI / RI
    # 如果CR小于0.1则通过，否则不通过
    CI = (maxEigVal - n) / (n - 1)
    CR = CI / RI[n]

    # 4.计算权重
    # 就是为的最大特征值所对应的特征向量中的各个元素标准化了，计算各个的百分比
    sumVec = np.sum(maxEigVec)
    weight = maxEigVec / sumVec  # 计算权重

    # 新增：检查权重向量中是否存在明显的虚部，实现了您的“验证”意图
    if np.any(np.abs(weight.imag) > 1e-10):  # 1e-10是一个很小的阈值，用于忽略计算噪声
        print("\n[警告]：权重计算结果中存在明显的虚部，请检查您的判断矩阵是否满足要求！")
        print(f"原始权重向量: {weight}\n")

    return {
        "weight": weight,
        "consistency_ratio": CR,
        "is_consistent": CR.real < 0.1
    }


def display_ahp_results(results, indices):
    """
    格式化并打印AHP分析的结果。

    参数:
    results (dict): 从 calculate_ahp_weights 函数返回的结果字典。
    indices (list): 与判断矩阵行列对应的指标名称列表。
    """
    cr_value = results["consistency_ratio"].real
    if results["is_consistent"]:
        print(f"一致性检验通过，值为{cr_value:.4f},小于0.1")
    else:
        print(f"一致性检验不通过,值为{cr_value:.4f},本应小于0.1")

    weight = results["weight"]
    # 原始的print(weight)被移到了上面的警告中，在有问题时才会打印，让正常输出更简洁

    # 美化输出各项指标，但是不进行纯实数部分处理是为了，如果出现虚数部分能够及时发现问题
    print("\n=== 各指标权重 ===")
    for i, (name, w) in enumerate(zip(indices, weight)):
        # 使用 .real 来获取复数的实部 进行显示，因为理论上权重应为实数
        print(f"{name}: 权重 = {w.real:.4f} ({w.real * 100:.2f}%)")

    print(f"\n权重总和: {np.sum(weight).real:.4f} (理论应为1)")


# 当这个脚本作为主程序运行时，执行以下代码
if __name__ == '__main__':
    # *****************************************************************(应该修改的地方)

    # 1.构造判断矩阵
    # 此处以身高，颜值，学历，性格来举例，修改的话修改判断矩阵
    A = np.array([[1, 3, 1, 1 / 3],
                  [1 / 3, 1, 1 / 2, 1 / 5],
                  [1, 2, 1, 1 / 3],
                  [3, 5, 3, 1]])  # 此处为判断矩阵

    # 定义与判断矩阵对应的指标名称
    indices_test = ["身高", "颜值", "学历", "性格"]

    # *****************************************************************(其他代码具有复用性，无需修改)

    try:
        # 调用函数执行AHP计算
        ahp_results = calculate_ahp_weights(A)
        # 调用函数显示结果
        display_ahp_results(ahp_results, indices_test)
    except ValueError as e:
        # 如果矩阵验证失败，则打印错误信息
        print(f"计算出错: {e}")
    except IndexError:
        # 如果矩阵维度超过RI表范围，则提示
        print(f"计算出错: 判断矩阵的维度（{A.shape[0]}）超出了RI表的范围，RI表最大支持9阶矩阵。")