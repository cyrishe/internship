import torch
import pandas as pd
import numpy as np
from FlagEmbedding import BGEM3FlagModel

vs_t = 0.82


class TSVProcessor:
    def __init__(self, filepath, sep='\t'):
        self.filepath = filepath
        self.sep = sep
        self.data = []
        self.headers = []
        self._read_tsv()

    def _read_tsv(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.headers = lines[0].strip().split(self.sep)
        for line in lines[1:]:
            fields = line.strip().split(self.sep)
            if len(fields) == len(self.headers):
                self.data.append(dict(zip(self.headers, fields)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_column(self, col_name):
        return [row[col_name] for row in self.data if col_name in row]

# 禁用网络，模型仅本地加载
model = BGEM3FlagModel(model_name_or_path='C:\\Users\\schoology\\Desktop\\shanghai\\1ssst\\models\\models\\bge-m3', use_fp16=True)

tsv_path = "test_data.tsv"  # 替换为你的TSV文件路径
tsv = TSVProcessor(tsv_path)

# 假设TSV有 '测试问题' 和 '原始问题' 两列
sentence1_list = [row['测试问题'] for row in tsv]
sentence2_list = [row['原始问题'] for row in tsv]
ans = [row['答案'] for row in tsv]

for i in range(len(sentence2_list)):
    sentence2_list[i] = f"{sentence2_list[i]}\n{ans[i]}" if ans[i] else sentence2_list[i]

# 计算embedding
embeddings_1 = model.encode(sentence1_list, batch_size=12, max_length=100)['dense_vecs']
embeddings_2 = model.encode(sentence2_list, batch_size=12, max_length=100)['dense_vecs']

# 计算相似度
def normalize(v):
    return v / (torch.norm(v, dim=1, keepdim=True) if isinstance(v, torch.Tensor) else np.linalg.norm(v, axis=1, keepdims=True))

embeddings_1 = np.array(embeddings_1)
embeddings_2 = np.array(embeddings_2)
similarities = (normalize(embeddings_1) * normalize(embeddings_2)).sum(axis=1)

is_passed = []
for i in range(len(similarities)):
    if similarities[i] >= vs_t:
        is_passed.append(1)
    else:
        is_passed.append(0)

# 保存结果
results = []
for i, row in enumerate(tsv):
    result_row = dict(row)
    result_row['similarity'] = similarities[i]
    results.append(result_row)

df = pd.DataFrame(results)
df['is_passed'] = is_passed
df.to_excel("tsv_results_mix.xlsx", index=False)
print("已保存到 tsv_results_mix.xlsx")