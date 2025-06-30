import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import re

MODEL_PATH = "./bert_manual_model"
TEST_CSV = "test.csv"
SUBMIT_CSV = "submission_with_keywords.csv"
MAX_LEN = 128
BATCH_SIZE = 1
TOP_K = 10  # 多选几个关键词，后面筛

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, output_attentions=True)
model.to(device)
model.eval()

class TweetDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return str(self.texts[idx])

df = pd.read_csv(TEST_CSV)
df.fillna("", inplace=True)
df['combined'] = df['text'] + " " + df['keyword'] + " " + df['location']
dataset = TweetDataset(df['combined'].tolist())

# 定义停用词（可根据需要增加）
STOPWORDS = set([
    'the', 'a', 'an', 'and', 'or', 'is', 'it', 'to', 'of', 'in', 'on', 'for', 'with', 'that', 'this',
    'we', 'are', 'be', 'by', 'as', 'at', 'from', 'was', 'were', 'but', 'not', 'you', 'i', 'he', 'she', 'they', 'them',
    'pad', 'sep'  # 新增这两个特殊token
])


def merge_subwords(tokens):
    words = []
    buffer = ''
    for token in tokens:
        if token.startswith("##"):
            buffer += token[2:]
        else:
            if buffer:
                words.append(buffer)
                buffer = ''
            buffer = token
    if buffer:
        words.append(buffer)
    return words

def clean_token(token):
    token = token.lower()
    token = re.sub(r'[^a-z0-9]', '', token)  # 去掉非字母数字
    return token

all_preds = []
all_keywords = []

with torch.no_grad():
    for text in tqdm(dataset, desc="Predicting with improved keywords"):
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        attentions = outputs.attentions

        pred = torch.argmax(logits, dim=1).cpu().item()
        all_preds.append(pred)

        # 取最后一层第一个head的attention权重
        last_attn = attentions[-1][0][0]  # [seq_len, seq_len]
        cls_attn = last_attn[0]            # CLS 对所有token的attention

        # 排除[CLS],取TOP_K+5，因为有停用词和特殊token需要剔除
        topk = torch.topk(cls_attn[1:MAX_LEN], k=TOP_K+5)
        topk_indices = topk.indices + 1  # 补偿偏移，恢复真实index

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        selected_tokens = [tokens[idx] for idx in topk_indices.cpu().numpy()]
        
        # 先去掉特殊token
        selected_tokens = [tok for tok in selected_tokens if tok not in ['[PAD]', '[SEP]']]
        
        # 合并subwords
        merged_words = merge_subwords(selected_tokens)


        # 清洗停用词和特殊token
        filtered_words = []
        for w in merged_words:
            cw = clean_token(w)
            if cw and cw not in STOPWORDS:
                filtered_words.append(cw)

        # 保证关键词不多于TOP_K个
        keywords_final = filtered_words[:TOP_K]

        all_keywords.append(", ".join(keywords_final))

df_result = pd.DataFrame({
    "id": df["id"],
    "target": all_preds,
    "keywords": all_keywords
})
df_result.to_csv(SUBMIT_CSV, index=False)
print(f"✅ 关键词提取完毕，结果保存到 {SUBMIT_CSV}")
