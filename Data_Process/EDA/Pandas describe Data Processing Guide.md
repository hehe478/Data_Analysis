

# **从描述到决策：利用 pandas.describe() 进行高级数据预处理的战略指南**

## **引言：描述性统计的诊断能力**

在任何严谨的探索性数据分析（Exploratory Data Analysis, EDA）工作流程中，pandas.DataFrame.describe() 函数都是基础性的第一步 1。然而，将其仅仅视为一个简单的汇总工具，会极大地低估其价值。实际上，describe() 是一个强大且多层面的诊断仪器，能够揭示数据集的内在结构、质量问题以及对后续机器学习建模的适用性。

可以将 describe() 想象成“数据的私人叙述者” 1。它不只是罗列数字，而是讲述隐藏在数据集背后的故事。在EDA的宏大叙事中，describe() 提供的统计摘要如同“主菜前的开胃菜” 1，为整个数据质量调查提供了最初的线索，帮助我们迅速发现趋势、异常和数据的整体健康状况。

本报告的核心论点是：超越简单地阅读数字，深入理解这些统计量对于数据结构、质量和适用性的深层含义。这将使我们能够制定一个战略性的、而非被动反应式的数据预处理计划。本指南将系统性地阐述如何将 describe() 的输出转化为具体、可辩护的数据清洗和特征工程决策，从而为构建稳健、可靠的高性能机器学习模型奠定坚实的基础。

---

## **第一部分：数值型特征分析的精通之道：诊断**

本部分将系统性地剖析 describe() 对数值型数据的输出结果，将每一个统计指标与特定的数据质量信号关联起来。默认情况下，describe() 会分析DataFrame中所有的数值类型列（如 int、float 等）3。

### **1.1 解码统计快照**

在深入解读之前，首先需要清晰地理解 describe() 为数值列返回的每一个指标的含义。这些指标共同构成了一幅关于数据分布、集中趋势和离散程度的画像 4。

* count：非空（non-NaN）观测值的数量。  
* mean：算术平均值。  
* std：标准差，衡量数据点相对于平均值的离散程度。  
* min：最小值。  
* 25%：第一四分位数（Q1），表示25%的数据小于此值。  
* 50%：中位数（Q2），即第二四分位数，表示50%的数据小于此值。  
* 75%：第三四分位数（Q3），表示75%的数据小于此值。  
* max：最大值。

### **1.2 诊断数据健康：解读线索**

掌握了基础定义后，下一步便是从这些数字中解读出关于数据健康的深层线索。

#### **缺失值 (count)**

count 指标显示了每列中非空条目的数量 4。数据清洗的第一步，就是将每列的 count 值与DataFrame的总行数（通过 df.shape 或 len(df) 获取）进行比较。任何不一致都直接表明该列存在缺失值及其数量。

然而，更深层次的分析不止于此。如果发现多个相关列（例如 billing\_address、billing\_city 和 billing\_zip）具有完全相同的、低于总行数的 count 值，这不仅仅意味着它们各自有缺失值，更可能暗示着一种系统性的缺失模式。这表明缺失的不是零散的字段，而是整个“账单地址”信息块。这一发现将指导我们采取更全面的处理策略，比如整体删除这些信息缺失的行，而不是孤立地对每一列进行插补。这种分析方式从“缺失了什么”上升到了“为什么会缺失”。这一诊断直接触发了EDA中的“缺失值处理”阶段，需要运用如 df.isnull().sum() 进行确认，并采用 fillna() 或更高级的插补技术来解决 2。

#### **数据偏度 (mean vs. 50%)**

mean 是算术平均值，而 50%（中位数）是数据的中心点 4。这两者之间的关系是判断数据分布对称性的一个极其有效的即时指标。

* 若 $mean \\approx 50\\%$：表明数据分布大致对称。  
* 若 $mean \> 50\\%$：表明数据呈\*\*右偏（正偏）\*\*分布。这意味着存在一些高值异常点，将平均值向右侧拉离中位数 5。  
* 若 $mean \< 50\\%$：表明数据呈\*\*左偏（负偏）\*\*分布。这意味着存在一些低值异常点，将平均值向左侧拉离中位数。

关注偏度并非纯粹的统计学练习，它对模型性能有着直接影响。许多机器学习模型，特别是线性模型（如线性回归、逻辑回归）以及那些依赖同方差性假设的模型，在处理高度偏斜的数据时表现会显著下降 10。因此，通过 describe() 识别出偏度，不仅仅是一个统计观察，更是一个直接的警告，提示我们某些特征可能违反了模型的基本假设。这一发现前瞻性地标记出了需要进行数据变换的特征，以确保模型的有效性和性能。

#### **特征尺度与波动性 (std, min, max)**

std（标准差）衡量数据围绕平均值的离散程度，而 min 和 max 则定义了数据的取值范围 4。跨列比较这些值至关重要。如果一个特征（如 Salary，范围可能在50,000到200,000之间）的尺度和标准差与另一个特征（如 Years\_of\_Experience，范围在1到20之间）存在巨大差异，这就发出了一个明确的信号：需要进行特征缩放。

特征缩放的需求并非普遍适用，而是与所选的机器学习算法紧密相关。树模型（如决策树、随机森林）在很大程度上对特征的尺度不敏感 12。然而，对于**基于距离**的算法（如K近邻、K均值聚类、支持向量机）或使用**梯度下降**进行优化的算法（如线性回归、逻辑回归、神经网络）而言，未经缩放的特征是灾难性的 12。尺度较大的特征将在距离计算或梯度更新中占据主导地位，从而有效地压制了尺度较小特征的贡献。因此，当 describe() 的输出揭示了不同特征间的尺度差异时，如果计划使用上述算法，那么特征缩放就成为一个高优先级的任务。

#### **异常值检测 (四分位数, min, max)**

四分位数（25%, 50%, 75%）将排序后的数据分为四个相等的部分，而 min 和 max 则展示了数据的绝对极值 1。初步检查异常值的一个简单方法是观察这些值之间的间隔。如果 75% 和 max 之间，或者 min 和 25% 之间存在一个巨大的鸿沟，这强烈暗示了极端值的存在。例如，如果75%的房价低于50万美元，但最大值却是1000万美元，这便是一个明显的异常值信号。

describe() 不仅能暗示异常值的存在，它还为系统性的、非可视化的异常值检测方法——\*\*四分位距（Interquartile Range, IQR）\*\*法则——提供了所需的精确数值。输出中的 25% 和 75% 分别是Q1和Q3。这使得我们能够立即计算出数据的“可接受”范围，将模糊的怀疑转变为一个可量化、可复现的流程。这为后续的异常值处理奠定了坚实的基础 14。

---

## **第二部分：数值型特征的可行预处理：治理**

本部分提供了一个实用的操作手册，旨在解决第一部分中诊断出的问题，并附有代码模式和战略性建议。

### **2.1 基于IQR的稳健异常值处理框架**

从怀疑转向正式流程，IQR方法是统计学上检测异常值的标准技术 15。

1. **计算IQR**：可以直接利用 describe() 的输出进行计算：IQR \= df\['feature'\].describe()\['75%'\] \- df\['feature'\].describe()\['25%'\]。  
2. **定义边界**：使用经典的 $1.5 \\times IQR$ 法则来定义异常值的边界：  
   * 下边界: $lower\\\_bound \= Q1 \- 1.5 \\times IQR$  
   * 上边界: $upper\\\_bound \= Q3 \+ 1.5 \\times IQR$ 14  
3. **识别与处理**：根据计算出的边界来识别和过滤异常值。  
   * **战略决策**：对于识别出的异常值，有多种处理策略。如果是明显的录入错误，可以选择**移除**。如果这些值是真实但极端的观测，可以考虑**截断（Capping）**，即用上下边界值替换超出范围的值。此外，由于偏度和异常值常常相伴而生，进行数据变换也是一种有效的处理方式。

### **2.2 修正偏斜分布**

根据 mean 与 median 的对比诊断结果，应用数学变换来使数据分布更接近对称，以满足模型假设 10。

* **右偏数据 (mean \> median)**：应用能够压缩较大值、同时对较小值影响较小的变换。  
  * **对数变换 (Log Transformation)**：这是最常用的方法。使用 np.log1p 可以优雅地处理包含0的值，因为它计算的是 $log(1+x)$ 10。  
  * **其他变换**：平方根变换 (np.sqrt) 和立方根变换也是有效的备选方案 10。  
* **左偏数据 (mean \< median)**：应用能够拉伸左侧尾部的变换。  
  * **幂变换 (Power Transformation)**：例如平方 (x\*\*2) 或立方 (x\*\*3) 变换 10。  
* **验证**：关键的一步是，在应用变换后，应重新运行 .describe() 或 .skew() 来验证 mean 和 median 是否更加接近，以及偏度系数是否趋近于0。

### **2.3 战略性特征缩放**

基于 std、min 和 max 诊断出的特征尺度差异，应用缩放技术，确保所有特征在模型训练中贡献平等 12。

* **标准化 (Standardization / Z-score Scaling)**  
  * **概念**：将数据重新缩放，使其均值为0，标准差为1。其计算公式为：$z \= (x \- \\mu) / \\sigma$。  
  * **适用场景**：这是最常用的缩放方法，对异常值的敏感度低于归一化。当数据分布近似高斯分布时效果尤佳，但即使分布未知，它也是一个稳健的默认选择 13。在Python中，可使用 sklearn.preprocessing.StandardScaler 实现。  
* **归一化 (Normalization / Min-Max Scaling)**  
  * **概念**：将数据重新缩放至一个固定的范围，通常是 \`\`。其计算公式为：$X\_{norm} \= (X \- X\_{min}) / (X\_{max} \- X\_{min})$ 21。  
  * **适用场景**：当算法要求数据在有界区间内时（如某些神经网络的激活函数），或当需要保留数据中的0值时，归一化非常有用。然而，由于其计算直接依赖于 min 和 max 值，它对异常值极为敏感 12。在Python中，可使用 sklearn.preprocessing.MinMaxScaler 实现。  
* **验证**：缩放完成后，对变换后的数据再次调用 .describe()，应能观察到预期的结果：对于标准化数据，mean 约等于0，std 约等于1；对于归一化数据，min 等于0，max 等于1。

### **表1：从数值指标到可行洞察的速查表**

下表总结了如何将 describe() 的数值统计输出转化为具体的诊断和后续行动，为数据从业者提供了一个快速参考。

| describe() 指标 | 信号 (诊断) | 推荐的下一步行动 |
| :---- | :---- | :---- |
| count \< 总行数 | 存在缺失数据 | 使用 .isnull().sum() 深入调查；规划插补 (fillna) 或移除 (dropna) 策略。 |
| mean\!= 50% (中位数) | 数据分布偏斜 | 使用直方图/KDE图进行可视化确认；应用变换（对数、平方根、Box-Cox）。 |
| std 值大，各列 min/max 差异悬殊 | 特征尺度不一 | 若使用基于距离/梯度的模型，应用特征缩放（标准化或归一化）。 |
| max \>\> 75% 或 min \<\< 25% | 存在潜在异常值 | 从 25% 和 75% 计算IQR；应用 $1.5 \\times IQR$ 法则识别并处理异常值。 |

---

## **第三部分：从类别型特征中提取战略洞察：诊断**

本部分专注于 describe(include=\['object', 'category'\]) 的独特输出及其对预处理决策的深远影响 3。

### **3.1 类别型摘要解析**

当应用于类别型或对象（object）类型的列时，describe() 返回四个关键指标，它们共同描绘了这些特征的分布特性 3。

* count：非空条目的数量。  
* unique：不同类别的数量。  
* top：出现频率最高的类别（即众数）。  
* freq：最高频类别的出现次数。

### **3.2 关键诊断信号**

#### **基数评估 (unique)**

unique 的计数值直接反映了类别型特征的**基数（Cardinality）** 23。这是规划编码策略时最重要的单一指标。

* **低基数 (unique \< \~15)**：这类特征是简单编码方法（如独热编码）的理想候选者。  
* **高基数 (unique \> \~50)**：这类特征带来了挑战。直接使用独热编码会导致“维度灾难”，创建一个极宽且稀疏的数据集，这会降低模型性能并增加计算成本 24。

一个值得特别关注的极端情况是当 unique 的值等于 count 时。这意味着该列中的每一个值都是独一无二的，例如客户ID、订单号等。这样的特征在原始形式下不具备任何预测能力，因为它无法为模型提供可泛化的模式，本质上是噪音或索引。通过 describe() 识别出这种情况，数据科学家可以立即将该列标记为从特征集中排除，或用于更高级的特征工程（例如，作为连接其他数据表的键）。

#### **类别不平衡检测 (top, freq)**

top 是最常见的类别，而 freq 是它出现的次数 4。通过比较 freq 和 count，可以快速评估类别不平衡的程度。一个简单的比率 $freq / count$，即为数据集中由最主要类别所占的比例。

如果这个比率非常高（例如，0.95），则意味着95%的数据都属于同一个类别。这是一个重大的警示信号。一个简单的模型可能仅通过预测这个主要类别就能达到95%的准确率，但这在实际应用中是毫无价值的。describe() 提供的这个快速计算，量化了构建一个有偏见、无实效模型的风险，并强调了采用特殊技术来处理类别不平衡的必要性。这对于分类问题来说是一个至关重要的诊断步骤。

---

## **第四部分：类别型特征的高级编码与处理：治理**

本部分将基于第三部分的诊断结果，提供处理类别型特征的具体策略和决策框架。

### **4.1 类别编码决策指南**

根据 unique 诊断出的基数来选择合适的编码策略。

* **低基数特征 (unique 值较小)**  
  * **独热编码 (One-Hot Encoding)**：这是标准方法，通过 pd.get\_dummies 实现。它为每个类别创建一个新的二进制列，从而避免了在类别间引入不存在的序数关系 24。对于线性模型，需要注意“虚拟变量陷阱”，并使用 drop\_first=True 参数来避免多重共线性。  
* **高基数特征 (unique 值较大)**  
  * **计数（频率）编码 (Count/Frequency Encoding)**：用每个类别在数据集中出现的频率（或计数）来替代该类别。这种方法简单，并且如果频率与目标变量相关，它可以捕捉到类别的重要性 25。  
  * **目标编码 (Target Encoding)**：用该类别对应的目标变量的平均值来替代类别。这是一种非常强大的技术，因为它直接利用了目标信息，但同时也存在很高的数据泄露和过拟合风险。实施时需要格外小心，例如使用交叉验证或留出集进行编码 23。  
  * **特征哈希 (Feature Hashing / Hashing Trick)**：使用哈希函数将可能无限多的类别映射到固定数量的新特征上。这种方法速度快、内存效率高，但可能会发生哈希冲突（即不同的原始类别被映射到同一个新特征），从而可能丢失信息 25。

### **表2：类别编码决策矩阵**

下表为从业者提供了一个清晰的结构化指南，以应对选择编码方法时的常见困惑。它将 unique 计数与一系列可行的编码选项直接关联，并明确了每种方法的优缺点。

| 编码方法 | 最佳适用基数 (unique) | 优点 | 缺点 |
| :---- | :---- | :---- | :---- |
| 独热编码 | 低 (\< 15\) | 不引入序数关系；易于解释。 | 导致维度灾难；不适用于高基数。 |
| 标签/序数编码 | 任何基数的序数数据 | 只增加一列；保留顺序信息。 | 对名义数据引入错误的序数关系。 |
| 计数/频率编码 | 中到高 | 简单；能捕捉类别流行度。 | 可能导致冲突（频率相同）；频率不一定与目标相关。 |
| 目标编码 | 高 | 直接捕捉与目标的关系；预测能力强。 | 极易过拟合；存在数据泄露风险。 |
| 特征哈希 | 非常高 | 可控输出维度；内存高效。 | 可能发生哈希冲突；可解释性差。 |

### **4.2 缓解类别不平衡的策略**

针对 $freq / count$ 比率诊断出的不平衡问题，可以采用以下策略来构建更稳健的模型：

* **重采样 (Resampling)**：通过**过采样**少数类（如SMOTE算法）或**欠采样**多数类来平衡数据集。  
* **类别权重 (Class Weights)**：在模型训练时调整损失函数，对少数类的错分施加更大的惩罚。  
* **选择合适的评估指标**：放弃使用准确率，转向对不平衡数据更稳健的指标，如精确率（Precision）、召回率（Recall）、F1分数（F1-Score）或AUC-ROC。

---

## **第五部分：解读时间序列特征摘要**

本部分将探讨在 datetime64 类型的列上使用 describe() 的特殊之处，这是一个经常被忽视的功能。

### **5.1 时间序列摘要的双重性**

describe() 在处理 datetime64 列时表现出双重性。

* **类数值摘要（默认行为）**：Pandas在内部将 datetime64 对象视为数值（自Unix纪元以来的纳秒数）27。因此，默认的 describe() 输出会包含 mean、min、max 和四分位数，这些值都是有效的时间戳 3。值得注意的是，输出中不包含 std，因为标准差对于时间点来说没有实际意义 27。  
* **类类别摘要**：通过先将时间列转换为 object 类型（例如，df\['date\_col'\].astype(str).describe()），我们可以获得 unique、top 和 freq 等统计信息，这些信息可以回答不同类型的问题 27。

### **5.2 从时间统计中获得可行洞察**

#### **定义时间跨度 (min, max)**

默认 describe() 输出中的 min 和 max 值，清晰地标示了数据集中最早和最晚的时间戳 27。这对数据验证至关重要：这个时间范围是否与预期的数据收集周期相符？它还定义了分析的范围，并且对于规划时间序列交叉验证（例如，在较早的数据上训练，在较晚的数据上测试）是必不可少的。

#### **评估规律性与粒度 (unique, top, freq)**

对时间列使用类类别摘要，top 和 freq 可以揭示数据的采样频率。如果数据是每小时收集一次，那么许多时间戳的 freq 应该为1（如果时间戳唯一），或者等于同时测量的传感器数量。如果某个时间戳的 freq 异常高，这可能表示数据重复问题或某个特定的周期性事件。这一发现直接指导了是否需要进行**重采样（resampling）**，例如使用 df.resample('D').mean() 将小时数据聚合为日数据 28。

#### **指导特征工程**

min 和 max 定义了一个时间画布。这个画布激发了创建一系列能够捕捉周期性模式的新特征的灵感。知道数据跨越了数年，提示我们可以创建 year 特征；跨越数月，则可以创建 month 和 quarter 特征；存在每日数据，则可以创建 day\_of\_week、is\_weekend 等特征。这些特征通常比原始时间戳本身具有更强的预测能力，并且可以方便地通过Pandas的 .dt 访问器创建 28。describe() 的输出为我们构思这些关键的工程特征提供了必要的上下文。

---

## **第六部分：超越 describe()：构建整体EDA视角**

本部分将 describe() 置于其应有的位置，既肯定其强大功能，也承认其局限性。

### **6.1 认识局限性**

* **单变量视角**：describe() 的主要局限在于它提供的是\*\*单变量（univariate）\*\*统计——它孤立地分析每一列 2。它无法揭示特征*之间*的关系、相关性或交互作用。  
* **平均值可能掩盖双峰分布**：对于双峰或多峰分布，mean 和 median 可能会产生误导。一个具有两个明显数据簇的特征，其均值可能恰好落在两个簇之间的低密度区域，那里实际上并没有多少数据点。

### **6.2 前进之路：可视化与相关性分析**

describe() 的作用是提出假设，而接下来的步骤是通过可视化和统计检验来验证这些假设。

* **可视化**：  
  * 使用**直方图**和**核密度估计（KDE）图**来直观地验证由 mean 与 median 关系所暗示的偏度。  
  * 使用**箱形图**来直观地确认由四分位数和 min/max 所暗示的异常值 2。  
* **相关性分析**：  
  * 使用 df.corr() 计算数值特征间的皮尔逊相关系数，并利用 seaborn 的**热力图**进行可视化，以探索双变量关系。  
* **高级工具**：  
  * 可以简要提及一些自动化的EDA工具，如 pandas-profiling。这类工具在 describe() 的基础上构建，能够生成更全面的报告，包括可视化、相关性警告等 29。但必须强调，深刻理解手动分析过程是掌握数据科学核心技能的根本。

---

## **结论：从数字到叙事**

pd.describe() 并非探索性数据分析的终点，而是一个功能强大的起点。它提供了一个信息密集的摘要，如果解读得当，就能转化为整个数据预处理流程的战略路线图。通过学习阅读这些描述性统计数据所讲述的“故事”，数据从业者可以从一种被动的、零敲碎打式的清洗过程，转变为一种主动的、系统性的、有理有据的工作流程。这种转变最终将引导我们构建出更稳健、更可靠的机器学习模型，从而真正释放数据的价值。

#### **Works cited**

1. Understanding pandas.describe(). I understand that learning data science… | by Hey Amit, accessed October 23, 2025, [https://medium.com/@heyamit10/understanding-pandas-describe-9048cb198aa4](https://medium.com/@heyamit10/understanding-pandas-describe-9048cb198aa4)  
2. Steps for Mastering Exploratory Data Analysis | EDA Steps ..., accessed October 23, 2025, [https://www.geeksforgeeks.org/data-analysis/steps-for-mastering-exploratory-data-analysis-eda-steps/](https://www.geeksforgeeks.org/data-analysis/steps-for-mastering-exploratory-data-analysis-eda-steps/)  
3. pandas.DataFrame.describe — pandas 2.3.3 documentation, accessed October 23, 2025, [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)  
4. Pandas DataFrame describe() Method \- GeeksforGeeks, accessed October 23, 2025, [https://www.geeksforgeeks.org/pandas/python-pandas-dataframe-describe-method/](https://www.geeksforgeeks.org/pandas/python-pandas-dataframe-describe-method/)  
5. A Comprehensive Guide to Pandas df.describe() for Descriptive ..., accessed October 23, 2025, [https://llego.dev/posts/pandas-df-describe-statistics-numeric-columns/](https://llego.dev/posts/pandas-df-describe-statistics-numeric-columns/)  
6. pandas: Get summary statistics for each column with describe() | note.nkmk.me, accessed October 23, 2025, [https://note.nkmk.me/en/python-pandas-describe/](https://note.nkmk.me/en/python-pandas-describe/)  
7. What is df.describe()? And What are its advantages in Data Analysis? \- Kaggle, accessed October 23, 2025, [https://www.kaggle.com/getting-started/156857](https://www.kaggle.com/getting-started/156857)  
8. Pandas Data Cleaning Tutorial \- Medium, accessed October 23, 2025, [https://medium.com/@mayurkoshti12/pandas-data-cleaning-tutorial-2dfb5af7d4b3](https://medium.com/@mayurkoshti12/pandas-data-cleaning-tutorial-2dfb5af7d4b3)  
9. Data Cleaning Using Python Pandas \- Complete Beginners' Guide \- Analytics Vidhya, accessed October 23, 2025, [https://www.analyticsvidhya.com/blog/2021/06/data-cleaning-using-pandas/](https://www.analyticsvidhya.com/blog/2021/06/data-cleaning-using-pandas/)  
10. Handling With Highly Skewed Data Set \- Kaggle, accessed October 23, 2025, [https://www.kaggle.com/code/setu06/handling-with-highly-skewed-data-set](https://www.kaggle.com/code/setu06/handling-with-highly-skewed-data-set)  
11. LogTransformer — 1.8.3 \- Feature-engine, accessed October 23, 2025, [https://feature-engine.trainindata.com/en/1.8.x/user\_guide/transformation/LogTransformer.html](https://feature-engine.trainindata.com/en/1.8.x/user_guide/transformation/LogTransformer.html)  
12. What is Feature Scaling and Why is it Important? \- Analytics Vidhya, accessed October 23, 2025, [https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)  
13. How Feature Scaling Improves Model Training in Deep Learning \- wellsr.com, accessed October 23, 2025, [https://wellsr.com/python/how-feature-scaling-improves-model-training-in-deep-learning/](https://wellsr.com/python/how-feature-scaling-improves-model-training-in-deep-learning/)  
14. How to Find Outliers in Data: IQR, DBSCAN & Python Examples ..., accessed October 23, 2025, [https://builtin.com/data-science/how-find-outliers-examples](https://builtin.com/data-science/how-find-outliers-examples)  
15. Interquartile Range to Detect Outliers in Data \- GeeksforGeeks, accessed October 23, 2025, [https://www.geeksforgeeks.org/machine-learning/interquartile-range-to-detect-outliers-in-data/](https://www.geeksforgeeks.org/machine-learning/interquartile-range-to-detect-outliers-in-data/)  
16. Identifying and Handling Outliers in Pandas \- Kaggle, accessed October 23, 2025, [https://www.kaggle.com/code/yunasheng/identifying-and-handling-outliers-in-pandas](https://www.kaggle.com/code/yunasheng/identifying-and-handling-outliers-in-pandas)  
17. Dealing with high skewed data? A Practical Guide Part III | by León Andrés M. \- Medium, accessed October 23, 2025, [https://medium.com/@lamunozs/dealing-with-high-skewed-data-a-practical-guide-part-iii-19fc38a10a7c](https://medium.com/@lamunozs/dealing-with-high-skewed-data-a-practical-guide-part-iii-19fc38a10a7c)  
18. Log Transformation and visualizing it using Python | by Tarique Akhtar \- Medium, accessed October 23, 2025, [https://tariqueakhtar-39220.medium.com/log-transformation-and-visualizing-it-using-python-392cb4bcfc74](https://tariqueakhtar-39220.medium.com/log-transformation-and-visualizing-it-using-python-392cb4bcfc74)  
19. Log Transformation \- GeeksforGeeks, accessed October 23, 2025, [https://www.geeksforgeeks.org/data-science/log-transformation/](https://www.geeksforgeeks.org/data-science/log-transformation/)  
20. How to handle Skewed Distribution \- Kaggle, accessed October 23, 2025, [https://www.kaggle.com/code/aimack/how-to-handle-skewed-distribution](https://www.kaggle.com/code/aimack/how-to-handle-skewed-distribution)  
21. Data Normalization with Pandas \- GeeksforGeeks, accessed October 23, 2025, [https://www.geeksforgeeks.org/python/data-normalization-with-pandas/](https://www.geeksforgeeks.org/python/data-normalization-with-pandas/)  
22. python \- Summary of categorical variables pandas \- Stack Overflow, accessed October 23, 2025, [https://stackoverflow.com/questions/64223060/summary-of-categorical-variables-pandas](https://stackoverflow.com/questions/64223060/summary-of-categorical-variables-pandas)  
23. Handling Machine Learning Categorical Data with Python Tutorial ..., accessed October 23, 2025, [https://www.datacamp.com/tutorial/categorical-data](https://www.datacamp.com/tutorial/categorical-data)  
24. Categorical Data Encoding: 7 Effective Techniques, accessed October 23, 2025, [https://datasciencedojo.com/blog/categorical-data-encoding/](https://datasciencedojo.com/blog/categorical-data-encoding/)  
25. 4 ways to encode categorical features with high cardinality \- Towards Data Science, accessed October 23, 2025, [https://towardsdatascience.com/4-ways-to-encode-categorical-features-with-high-cardinality-1bc6d8fd7b13/](https://towardsdatascience.com/4-ways-to-encode-categorical-features-with-high-cardinality-1bc6d8fd7b13/)  
26. All you need to know about encoding techniques\! | by Indraneel Dutta Baruah \- Medium, accessed October 23, 2025, [https://medium.com/anolytics/all-you-need-to-know-about-encoding-techniques-b3a0af68338b](https://medium.com/anolytics/all-you-need-to-know-about-encoding-techniques-b3a0af68338b)  
27. python \- Pandas: describe() for datetime column \- Stack Overflow, accessed October 23, 2025, [https://stackoverflow.com/questions/76992453/pandas-describe-for-datetime-column](https://stackoverflow.com/questions/76992453/pandas-describe-for-datetime-column)  
28. How to handle time series data with ease — pandas 2.3.3 documentation \- PyData |, accessed October 23, 2025, [https://pandas.pydata.org/docs/getting\_started/intro\_tutorials/09\_timeseries.html](https://pandas.pydata.org/docs/getting_started/intro_tutorials/09_timeseries.html)  
29. Pandas Profiling (Exploratory Data Analysis) — Understand your data first\! | by Ethan Duong, accessed October 23, 2025, [https://medium.com/@ethan.duong1120/pandas-profiling-exploratory-data-analysis-understand-your-data-first-8b7e095c0d2b](https://medium.com/@ethan.duong1120/pandas-profiling-exploratory-data-analysis-understand-your-data-first-8b7e095c0d2b)