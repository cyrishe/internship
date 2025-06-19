import pandas as pd
import torch
from FlagEmbedding import FlagReranker

# 读取数据
df = pd.read_excel('sentence_rel.xlsx')
sentence1 = df['sentence1'].tolist()
sentence2 = df['sentence2'].tolist()
prelabel = df['label'].tolist()

# 初始化 Reranker 模型
reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)

# 构建句子对
pairs = list(zip(sentence1, sentence2))

# 模型计算相似度得分（raw logits）
logits = reranker.compute_score(pairs)

# 加上 sigmoid 映射，归一化到 0~1
similarities = [torch.sigmoid(torch.tensor(s)).item() for s in logits]
df['similarity'] = similarities

def confusion_cols(pred, label):
    """
    Returns two lists:
    - TF: 'TP', 'TN', 'FP', 'FN' for each prediction
    """
    tf_list = []
    for p, l in zip(pred, label):
        if p == 1 and l == 1:
            tf_list.append('TP')
        elif p == 0 and l == 0:
            tf_list.append('TN')
        elif p == 1 and l == 0:
            tf_list.append('FP')
        elif p == 0 and l == 1:
            tf_list.append('FN')
    return tf_list

for i in range(50):
    va_t = 0.5 + i / 100
    va = [1 if sim > va_t else 0 for sim in similarities]

    correct_predictions = sum(va[j] == prelabel[j] for j in range(len(prelabel)))
    accuracy = correct_predictions / len(prelabel)

    matches = [1 if va[j] == prelabel[j] else 0 for j in range(len(prelabel))]
    p_va = ["T" if v == 1 else "F" for v in va]
    p_prel = ["P" if p == 1 else "N" for p in prelabel]

    # Add confusion matrix columns
    tf_col, pn_col = confusion_cols(va, prelabel)
    df['TF'] = tf_col
    df['PN'] = pn_col

    df['label'] = va
    df['match'] = matches
    df['prelabel'] = prelabel
    df['C'] = p_va
    df['M'] = p_prel

    # 阈值边界估计
    left = max([similarities[j] for j in range(len(prelabel)) if va[j] == 0 and prelabel[j] == 1], default=0)
    right = min([similarities[j] for j in range(len(prelabel)) if va[j] == 1 and prelabel[j] == 0], default=1)

    print(f"Threshold: {va_t:.4f}, Accuracy: {accuracy * 100:.2f}%, Left: {left:.4f}, Right: {right:.4f}")

    df.to_excel(f"sentence_threshold={va_t:.4f}_accuracy={accuracy:.4f}.xlsx", index=False)
