import numpy as np

# 层次分析法(Analytic Hierarchy Process)，用于计算权重
# 本代码可复用，如需修改，仅需修改判断矩阵部分 即第1部分
RI = np.array([0,0,0,0.58,0.90,1.12,1.24,1.32,1.41,1.45])  # 导入RI，后面用于一致性检验
# 特地多打了一个0，为的是能够和数组维度精准匹配

# *****************************************************************(应该修改的地方)

# 1.构造判断矩阵
# 此处以身高，颜值，学历，性格来举例，修改的话修改判断矩阵
A = np.array([[1,3,1,1/3],
              [1/3,1,1/2,1/5],
              [1,2,1,1/3],
              [3,5,3,1]])  # 此处为判断矩阵

# *****************************************************************(其他代码具有复用性，无需修改)

# 2.计算特征值和特征向量，找到最大特征值与其所对应的特征向量
n , m = A.shape # 获取矩阵的维度，只是取了两个不同变量名，实际上m,n数值是一致的
# 验证是否为正互反矩阵
# 因为若是输入的矩阵过大或者复杂可能存在一定错误，所以进行一定判断保证输入矩阵的正确性
for i in range(n):
    if not np.isclose(A[i, i], 1):
        raise ValueError(f"判断矩阵对角线元素必须为1，第{i + 1}行第{i + 1}列错误")
    for j in range(i + 1, n):
        if not np.isclose(A[i, j], 1 / A[j, i]):
            raise ValueError(f"判断矩阵不满足互反性，第{i + 1}行第{j + 1}列与第{j + 1}行第{i + 1}列错误")
    if np.any(A[i, :] <= 0):
        raise ValueError(f"判断矩阵元素必须为正数，第{i + 1}行存在非正数")

# 开始获取特征值与特征向量
eigenvalues , eigenvectors = np.linalg.eig(A)  # 计算特征值和特征向量
maxEigValIndex = np.argmax(eigenvalues)  # 最大特征值对应索引
maxEigVal = eigenvalues[maxEigValIndex]  # 最大特征值
maxEigVec = eigenvectors[:,maxEigValIndex]  # 最大特征值所对应的特征向量,注意此处是列，不是行，应该取列

# 3.一致性检验
# CI = (maxEigVal - n）/（n - 1），最大特征值和矩阵的维度
# CR = CI / RI
# 如果CR小于0.1则通过，否则不通过
CI = (maxEigVal - n)/(n - 1)
CR = CI / RI[n]
if CR > 0.1:
    print(f"一致性检验不通过,值为{CR:.4f},本应小于0.1")
else:
    print(f"一致性检验通过，值为{CR:.4f},小于0.1")

# 4.计算权重
# 就是为的最大特征值所对应的特征向量中的各个元素标准化了，计算各个的百分比
sumVec = np.sum(maxEigVec)
weight = maxEigVec / sumVec  # 计算权重
print(weight)
# 美化输出各项指标，但是不进行纯实数部分处理是为了，如果出现虚数部分能够及时发现问题
print("\n=== 各指标权重 ===")
indices = ["身高", "颜值", "学历", "性格"]  # 与判断矩阵列对应
for i, (name, w) in enumerate(zip(indices, weight)):
    print(f"{name}: 权重 = {w:.4f} ({w * 100:.2f}%)")

print(f"\n权重总和: {np.sum(weight):.4f} (理论应为1)")