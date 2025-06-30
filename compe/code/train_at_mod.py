import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

class BERTClassifier(nn.Module):
    def __init__(self, pretrained="bert-base-uncased", num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]  # CLS
        return self.classifier(cls_output)

# === 数据加载 ===
df = pd.read_csv("train.csv").fillna("").head(1000)
df["text_input"] = df["keyword"] + " " + df["location"] + " " + df["text"]
texts = df["text_input"].tolist()
labels = df["target"].tolist()

# === 数据集和加载器 ===
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
dataset = TweetDataset(texts, labels, tokenizer, max_len=128)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# === 模型、优化器 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# === 训练 ===
model.train()
for epoch in range(3):
    losses, preds, trues = [], [], []
    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        trues.extend(labels.cpu().numpy())

    acc = accuracy_score(trues, preds)
    print(f"✅ Epoch {epoch+1} - Loss: {sum(losses)/len(losses):.4f} - Accuracy: {acc:.4f}")

# === 保存模型 ===
torch.save(model.state_dict(), "bert_tweet_classifier.pt")


# === 关键词提取（基于 attention） ===
model.eval()
with torch.no_grad():
    # 示例推文（取第1条）
    sample_text = texts[0]
    encoded = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=128)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # 获取 BERT 的所有中间输出
    output = model.bert(**encoded, output_attentions=True)
    attentions = output.attentions  # list of (batch, heads, seq_len, seq_len)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    # === 提取最后一层，取 CLS 对其他 token 的注意力（平均头） ===
    last_layer = attentions[-1]            # (1, heads, seq_len, seq_len)
    cls_attn = last_layer[0, :, 0, :]      # (heads, seq_len) CLS -> token
    mean_attn = cls_attn.mean(dim=0)       # (seq_len,)

    # === 提取排名前几的关键词（排除 [CLS], [SEP], [PAD]） ===
    keywords = []
    for token, score in zip(tokens, mean_attn):
        if token not in ("[CLS]", "[SEP]", "[PAD]"):
            keywords.append((token, score.item()))

    keywords = sorted(keywords, key=lambda x: -x[1])[:5]
    print(f"\n📌 示例推文：{sample_text}")
    print(f"✅ 关键词提取结果：{[w for w, _ in keywords]}")

    # === 保存到文件 ===
    import json, os
    os.makedirs("output", exist_ok=True)
    with open("output/keywords.json", "w") as f:
        json.dump(keywords, f, indent=2)
    print("✅ 关键词保存成功：output/keywords.json")
