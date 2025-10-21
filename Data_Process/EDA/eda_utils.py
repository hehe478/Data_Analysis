import pandas as pd
import numpy as np  # 导入 numpy 以便更稳健地处理 dtypes


def inspect_and_summarize_dataframe(csv_path: str, columns_to_ignore_unique: list = None):
    """
    加载 CSV 文件并执行一个全面的初步数据侦察和统计摘要。

    此函数涵盖：
    1. 基础侦察 (shape, head, tail, info)
    2. 针对数值型和对象/布尔型的描述性统计
    3. 遍历每种数据类型，输出：
       - 该类型的 describe() 摘要
       - 对非数值型列，输出 value_counts() (包含缺失值)
       - 对非数值型列，输出 unique() (可跳过指定列)

    参数:
    ----------
    csv_path : str
        需要加载和分析的 CSV 文件的路径。

    columns_to_ignore_unique : list, 可选
        一个字符串列表，包含不希望打印 .unique() 结果的列名。
        (例如 ['Name', 'ID'] 这种基数非常高的列)。
        默认值为 None，将处理为空列表 [].
    """

    # 初始化要忽略的列
    if columns_to_ignore_unique is None:
        columns_to_ignore_unique = []

    # --- 1.1 初步数据侦察：加载与宏观评估 ---
    try:
        df = pd.read_csv(csv_path)
        print(f"===== 正在加载数据: {csv_path} =====")
    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径是否正确: {csv_path}")
        return
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return

    print("\n--- 1.1.1 数据维度 (Shape) ---")
    print(df.shape)

    print("\n--- 1.1.2 数据头部 (Head) ---")
    print(df.head())

    print("\n--- 1.1.3 数据尾部 (Tail) ---")
    print(df.tail())

    print("\n--- 1.1.4 结构总览 (Info) ---")
    print(df.info())

    print("\n" + "=" * 60 + "\n")

    # --- 1.2 描述性统计摘要 ---
    print("--- 1.2.1 数值型特征摘要 (Describe - Numeric) ---")
    # 默认只包含数值型
    print(df.describe())

    print("\n--- 1.2.2 类别型特征摘要 (Describe - Object/Bool) ---")
    # 显式指定 'object' 和 'bool'
    try:
        print(df.describe(include=['object', 'bool']))
    except ValueError:
        print("数据中没有 'object' 或 'bool' 类型的列。")

    print("\n" + "=" * 60 + "\n")

    # --- 1.2.3 按数据类型进行深度摘要 (Value Counts & Unique) ---

    # 1. 获取所有列的 dtype
    all_dtypes = df.dtypes
    print("所有列的数据类型：")
    print(all_dtypes)
    print("\n" + "=" * 60 + "\n")

    # 2. 提取唯一的 dtype（去重）
    unique_dtypes = all_dtypes.unique()

    # 3. 逐个 dtype 处理
    for dtype in unique_dtypes:
        print(f"===== 数据类型：{dtype} 的统计摘要 =====")

        # 3.1 输出该类型列的 describe() 结果
        try:
            desc = df.describe(include=[dtype])
            print(desc)
        except ValueError:
            print(f"没有可描述的 '{dtype}' 类型列。")

        print("\n" + "-" * 30 + "\n")  # 分隔

        # 3.2 筛选出该类型的所有列，逐个输出 value_counts()

        # 使用 numpy 的 is_numeric_dtype 来更稳健地判断是否为数值型
        if np.issubdtype(dtype, np.number):
            print(f"类型 {dtype} 是数值型，跳过 value_counts() 和 unique()。")
            print("\n" + "=" * 60 + "\n")
            continue

        # 获取该类型的所有列名
        cols_of_dtype = df.select_dtypes(include=[dtype]).columns

        for col in cols_of_dtype:
            print(f"--- 列 {col} 的值计数 (Value Counts) ---")
            try:
                # dropna=False 显示缺失值(NaN)的数量
                print(df[col].value_counts(dropna=False))
            except Exception as e:
                print(f"计算 {col} 的 value_counts 时出错: {e}")

            print("\n" + "-" * 20 + "\n")  # 分隔列之间的输出

            # 如果列在“忽略列表”中，则跳过 .unique() 的打印
            if col in columns_to_ignore_unique:
                print(f"已跳过打印列 {col} 的 .unique() 值。")
                print("\n" + "-" * 20 + "\n")
                continue

            print(f"--- 列 {col} 的唯一值 (Unique) ---")
            try:
                print(df[col].unique())
            except Exception as e:
                print(f"获取 {col} 的 unique 值时出错: {e}")

            print("\n" + "-" * 20 + "\n")  # 分隔列之间的输出

        print("\n" + "=" * 60 + "\n")  # 分隔不同 dtype 之间的输出

# --- 如何使用这个函数 ---

# 假设你的泰坦尼克号 'train.csv' 文件在同一个文件夹下
# 'Name' 列的唯一值太多，打印出来意义不大，所以我们忽略它

print("===== 开始分析泰坦尼克号数据集 =====")
inspect_and_summarize_dataframe(
    csv_path='train.csv',
    columns_to_ignore_unique=['Name', 'Ticket']
)

# 假设你还有一个 'test.csv'
# print("\n\n===== 开始分析测试数据集 =====")
# inspect_and_summarize_dataframe(csv_path='test.csv')