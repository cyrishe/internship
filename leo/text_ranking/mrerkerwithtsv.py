import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from FlagEmbedding import FlagReranker
import torch

va_t = 0.82  # 阈值
va_o = 10  # 偏移值

df = pd.read_csv("test_data.tsv", sep="\t")

query = df['测试问题']
document = df['原始问题']
answer = df['答案']

for i in range(len(query)):
    document[i] = document[(i+va_o)%len(document)] + '\n' + answer[(i+va_o)%len(document)]


# 加载模型和分词器
model_name = Path('C:\\Users\\schoology\\Desktop\\shanghai\\1ssst\\models\\bge-reranker')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = FlagReranker(model_name, use_fp16=True)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 示例输入（query, document）

# 拼接成输入格式：[query, document]
pairs = list(zip(query, document))
logits = model.compute_score(pairs)


similarities = [torch.sigmoid(torch.tensor(s)).item() for s in logits]
df['similarity'] = similarities

df['偏移后结果'] = document

# 输出得分（越高越相关）
score = similarities
df['similarity'] = similarities

th = []
for i in range(len(score)):
    if score[i] > va_t:
        th.append(1)
    else:
        th.append(0)
df['label'] = th
# 保存结果到新的 xlsx 文件
df.to_excel("test_data_with_scores_mix_negative_result.xlsx", index=False)