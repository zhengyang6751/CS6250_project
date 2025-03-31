# -*- coding: utf-8 -*-
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. 加载数据
csv_file_path = '/Users/zhengyang/cs6250/project/combined_data.csv'
data = pd.read_csv(csv_file_path)

# 2. 数据预处理
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # 移除特殊字符
    return str(text)

data['comments'] = data['comments'].fillna('').apply(clean_text)

# 强制转换标签列为 int
data['contains_slash_s'] = data['contains_slash_s'].astype(int)

print("标签分布:", data['contains_slash_s'].value_counts())

# 3. 令牌化
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

dataset = TextDataset(data['comments'], data['contains_slash_s'])

# 4. 训练参数
batch_size = 16
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 6. 训练
model.train()
for epoch in range(3):  # 训练3个epoch
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss  # Hugging Face模型已自动计算损失
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

# 7. 评估
def evaluate_model(model, dataset):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

evaluate_model(model, dataset)
