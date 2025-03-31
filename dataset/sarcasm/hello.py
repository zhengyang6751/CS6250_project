import pandas as pd

# 加载数据集
csv_file_path = '/Users/zhengyang/cs6250/project/combined_data.csv'
data = pd.read_csv(csv_file_path)

# 检查标签分布
label_counts = data['contains_slash_s'].value_counts()
print("标签分布:")
print(label_counts)

# 计算比例
total_samples = len(data)
print(f"正类样本比例: {label_counts[1] / total_samples:.2%}")
print(f"负类样本比例: {label_counts[0] / total_samples:.2%}")