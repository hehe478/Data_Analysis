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
        (例如 ['Name', 'ID'] 这种基数非常高 的列)。
        默认值为 None，将处理为空列表 [].
    """

    # 初始化要忽略的列
    if columns_to_ignore_unique is None:
        columns_to_ignore_unique = []

    # # 1. 设置显示的最大列数（None 表示“无限”）
    # pd.set_option('display.max_columns', None)
    # # 2. 设置显示的最大行数（None 表示“无限”）
    # # （这个在你使用 .T 时非常有用）
    # pd.set_option('display.max_rows', None)
    # # 3. 你还可以设置每行的最大宽度，防止换行
    # pd.set_option('display.width', 1000)

    # 这些可以用于调整显示，即
    # C22 C26          3
    #               ...
    # E34              1
    # 如这种在控制台的省略输出添加了这些代码可以全部输出
    # 但是一般不会这么使用，人眼查看所有的数据无意义，当然也可以将这些结果导出csv中进行记录

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
    print(df.shape)  # 可以用来了解数据规模
    # 数据少可能无法用来训练较为复杂的深度学习模型，得出的统计结论也可能不那么具有普遍性
    # 数据多需要考虑是否使用分块，抽样处理
    # 特征是否相对于样本数量 来说有点过多了，可能导致过拟合，影响是否进行特征选择或者降维
    # 了解数据是否完全加载，是否因部分原因导致丢失，例如分隔符的错误

    print("\n--- 1.1.2 数据头部 (Head) ---")
    print(df.head())
    # 验证数据是否加载正确，例如列名是否都正确加载了，索引是否正确
    # 让你有机会检查数据格式，例如日期是否需要转化，str数据是否需要经过转化成数字等

    print("\n--- 1.1.3 数据尾部 (Tail) ---")
    print(df.tail())
    # 检查文件末尾是否有垃圾数据，例如人为 的合计，数据来源等数据信息
    # 检查数据是否加载正确，例如shape说总共有10000行，那tail()的最后的索引应该是9999

    print("\n--- 1.1.4 结构总览 (Info) ---")
    print(df.info(verbose = True,memory_usage='deep'))  # verbose 用来保证如果太多列了即太多特征了，他不会把中间的各列的信息给忽略不打印
    # verbose这个参数最常用在例如机器学习中展示进度，加载函数中显示加载日志，以及此处，取值0(False)，1(True)，2
    # 不加memory_usage = 'deep' 导致只是计算指针在内存中的占用空间而不是指针的指向的对象的占用空间，不准确，适用于存在大量object的时候
    # 可以用来查看是否有的列存在字符串导致看似是数值实际是字符串类型，比如年龄，本应是数字，却被识别为object，这就说明其中存在str
    # 可以用来帮你快速了解哪些列存在Nan
    # 评估内存占用，如果发现内存过大需要优化，可以将一些object例如性别转化为category，节约一下

    print("\n" + "=" * 60 + "\n")

    # **************************************************************************************************(具体分析是否取消注释)

    # describe的数据如何使用具体见另一文件Pandas describe Data Processing Guide.md

    # **************************************************************************************************(具体分析是否取消注释)

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

        # **************************************************************************************************(具体分析是否取消注释)


        # # 使用 numpy 的 is_numeric_dtype 来更稳健地判断是否为数值型
        # if np.issubdtype(dtype, np.number):
        #     print(f"类型 {dtype} 是数值型，跳过 value_counts() 和 unique()。")
        #     print("\n" + "=" * 60 + "\n")
        #     continue


        # **************************************************************************************************(具体分析是否取消注释)
        # 如果不存在低基数类型数据可以取消注释，高基数类型，属于数值型的，量很多，离散或连续的，unique没有意义
        # 但是低基数的可以看看unique，因为其实本质上也是类型数据,例如星期1，2，3，4等等

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
            # 打印unique可以用于识别健全行检查，尤其是object类型，例如打印出了['man','male','男','女','female']就需要进行数据清洗
            # 了解取值范围，例如学历的['本科','硕士','博士']就不能用独热编码，就可能要考虑标签编码例如:1,2,3
            # 哪些建议用unique，例如类别型数据，object类，boolean类的可以看看；低基数类型的数值型可以看看(星期1,2,3,4,5等，本质也是分类)
            # 哪些不建议用unique,高基数数据，连续的数值没必要看，建议看describe；用户ID之类的唯一标识没必要，但是可以设为index方便索引
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