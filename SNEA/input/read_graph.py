import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
df = pd.read_csv("Epinions.csv")

# 去除包含 NaN 的行
df = df.dropna(subset=['id1', 'id2', 'sign'])

# 计算节点数量（通过ID的最大值+1）
num_nodes = max(df['id1'].max(), df['id2'].max()) + 1

# 划分训练集和测试集，按 80% 训练集，20% 测试集的比例划分
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 保存训练集文件
train_edges = len(train_df)
with open("Epinions_train.txt", "w") as f_train:
    f_train.write(f"{num_nodes} {train_edges}\n")  # 第一行是节点数量和边数量
    for _, row in train_df.iterrows():
        if pd.notna(row['id1']) and pd.notna(row['id2']) and pd.notna(row['sign']):
            f_train.write(f"{int(row['id1'])} {int(row['id2'])} {int(row['sign'])}\n")  # 写入每条边

# 保存测试集文件
test_edges = len(test_df)
with open("Epinions_test.txt", "w") as f_test:
    f_test.write(f"{num_nodes} {test_edges}\n")  # 第一行是节点数量和边数量
    for _, row in test_df.iterrows():
        if pd.notna(row['id1']) and pd.notna(row['id2']) and pd.notna(row['sign']):
            f_test.write(f"{int(row['id1'])} {int(row['id2'])} {int(row['sign'])}\n")  # 写入每条边
