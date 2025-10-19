import pandas as pd


df = pd.read_csv('train.csv')
print(df.shape)
print(df.head())  # 请务必记住这是一个方法，所以必须加括号，否则print(df.head)将输出方法的元信息
print(df.tail())  # 同上
print(df.info())
print(df.describe())
print(df.describe(include=['object', 'bool']))
# print(df.describe(include=['datetime64']))
# df['交易日期'].dt.year.value_counts()
# print(df.describe(include=['timedelta64']))
print(df['Fare'].value_counts())