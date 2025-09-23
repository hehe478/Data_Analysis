import numpy as np
# Topsis法

#  极小型转化
def minTomax(maxx,x):
    x = list(x)
    x = [[maxx - i] for i in x]
    return np.array(x)

# 中间型转化
def midTomax(midx,x):
    x = list(x)
    x = [abs(i - midx) for i in x]
    M = max(x)
    if M == 0:
        M = 1
    x = [[(1 - i / M)] for i in x]
    return np.array(x)

# 区间型转化
def regTomax(lowx,highx,x):
    x = list(x)
    M = max(lowx - min(x), max(x) - highx)
    if M == 0:
        M = 1
    ans = []
    for i in range(len(x)):
        if x[i] < lowx:
            ans.append([(1 - (lowx - x[i]) / M)])
        elif x[i] > highx:
            ans.append([(1 - (x[i] - highx) / M)])
        else:
            ans.append([1])
    return np.array(ans)


# 主函数调用
print("请输入参评数目：") #例如3个人
n = int(input()) #例如四项美德
print("请输入指标数目：")
m = int(input())

print("请输入类型矩阵：1.极大型 2.极小型 3.中间型 4.区间型")
kind = input().split(" ")

# 接受矩阵转化为numpy数组
print("请输入矩阵：")
matrix = np.array([list(map(float, input().split())) for _ in range(n)]).reshape(n, m)
print(matrix)

# 统一指标，转化为极大型
x = np.zeros(shape = (n,1))
for i in range(m):
    if kind[i] == "1":
        v = np.array(matrix[:,i])
    elif kind[i] == "2":
        maxX = np.max(matrix[:,i])
        v = minTomax(maxX , matrix[:,i])
    elif kind[i] == "3":
        print("类型三：请输入最优值")
        maxX = eval(input())
        v = midTomax(maxX , matrix[:,i])
    elif kind[i] == "4":
        print("类型四：请输入区间最小值")
        lowA = eval(input())
        print("类型四：请输入区间最大值")
        highA = eval(input())
        v = regTomax(lowA,highA,matrix[:,i])
    if i == 0:
        X = v.reshape(-1,1)
    else:
        X = np.hstack([X , v.reshape(-1,1)])
print("统一后的矩阵为\n{}".format(X))

# 对统一后的矩阵进行标准化处理
X = X.astype(float)
for j in range(m):
    X[:,j] = X[:,j]/np.sqrt(sum(X[:,j]**2))
print("标准化矩阵为\n{}".format(X))

# 计算最大最小距离
maxD = np.max(X, axis=0)
minD = np.min(X, axis=0)
d_z = np.sqrt(np.sum(np.square((X - np.tile(maxD, (n, 1)))), axis=1))
d_y = np.sqrt(np.sum(np.square((X - np.tile(minD, (n, 1)))), axis=1))

print("每个指标最大值：",maxD)
print("每个指标最小值：",minD)
print("d+向量：",d_z)
print("d-向量：",d_y)

# 计算得分
s = d_y / (d_z + d_y)
Score = 100 * s/sum(s)
for i in range(len(Score)):
    print("第{}个人得分为：{}".format(i+1,Score[i]))
