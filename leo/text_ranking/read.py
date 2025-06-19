import pandas as pd
import numpy as np
from FlagEmbedding import BGEM3FlagModel

df = pd.read_excel("sentence_rel.xlsx")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

sentence1 = df['sentence1'].tolist()
sentence2 = df['sentence2'].tolist()
prelabel = df['label'].tolist()


def normalize(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)

# Calculate embeddings for all sentences first
embeddings_1_all = model.encode(sentence1, batch_size=12, max_length=100)['dense_vecs']
embeddings_2_all = model.encode(sentence2)['dense_vecs']
similarities = normalize(embeddings_1_all) @ normalize(embeddings_2_all).T
similarities = similarities.diagonal()  # Get the similarity scores for each pair

# Try different thresholds
for i in range(50):  # Try 50 different thresholds
    va_t = 0.5 + i/100 #va_t就是阈值
    va = []
    p_va=[]
    p_prel=[]
    
    # Calculate labels for current threshold
    for similarity in similarities:
        va.append(1 if similarity > va_t else 0)
    
    # Calculate accuracy
    correct_predictions = sum(1 for i in range(len(prelabel)) if va[i] == prelabel[i])
    accuracy = correct_predictions / len(prelabel)
    
    # Calculate matches (1 if predicted label matches pre-defined label, 0 otherwise)
    matches = [1 if va[i] == prelabel[i] else 0 for i in range(len(prelabel))]
    
    # Update dataframe
    df['similarity'] = similarities
    df['prelabel'] = prelabel
    df['label'] = va
    for i in range(len(va)):
        p_va.append("T" if va[i]==1 else "F")
        p_prel.append("P" if prelabel[i] == 1 else "N")
    df['match'] = matches
    df['C'] = p_va
    df['M'] = p_prel
    
    # Save results for current threshold
    df.to_excel("sentence_rel_similarity_pass=%.4f_similarity=%.4f.xlsx" % (va_t, accuracy), index=False)
    
    # Calculate threshold bounds
    left = float(0)
    right = float(1)
    for i in range(len(prelabel)):
        if va[i] == 0 and prelabel[i] == 1:
            left = max(left, similarities[i])
        elif va[i] == 1 and prelabel[i] == 0:
            right = min(right, similarities[i])
    
    print("Threshold: %.4f, Accuracy: %.4f%%, Left: %.4f, Right: %.4f" % (va_t, accuracy * 100, left, right))
    