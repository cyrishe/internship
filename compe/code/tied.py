import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

#   加载测试数据
test_df = pd.read_csv('test.csv')
test_df['location'] = test_df['location'].fillna('')
test_df['keyword'] = test_df['keyword'].fillna('')
test_df['text'] = test_df['text'].fillna('')

#   加载处理器
with open('tokenizer_text.pkl', 'rb') as f:
    tokenizer_text = pickle.load(f)
with open('location_le.pkl', 'rb') as f:
    location_le = pickle.load(f)
with open('keyword_le.pkl', 'rb') as f:
    keyword_le = pickle.load(f)

#   安全编码函数（避免 unseen label 报错）
def safe_label_encode(series, label_encoder, unknown_value=0):
    label2id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    encoded = []
    for val in series:
        val = '' if pd.isna(val) else val
        encoded.append(label2id.get(val, unknown_value))
    return np.array(encoded)

#   文本处理
MAX_LEN_TEXT = 50
sequences_test_text = tokenizer_text.texts_to_sequences(test_df['text'])
padded_test_text = pad_sequences(sequences_test_text, maxlen=MAX_LEN_TEXT, padding='post', truncating='post')

#   location 和 keyword 编码
test_location_encoded = safe_label_encode(test_df['location'], location_le)
test_keyword_encoded = safe_label_encode(test_df['keyword'], keyword_le)

#   加载模型
model = load_model('twitter_model.h5')

#   执行预测
pred_probs = model.predict({
    'text_input': padded_test_text,
    'location_input': test_location_encoded,
    'keyword_input': test_keyword_encoded
})
pred_labels = (pred_probs >= 0.5).astype(int).flatten()

#   输出 CSV
submission = pd.DataFrame({
    'id': test_df['id'],
    'target': pred_labels
})
submission.to_csv('submission_ans.csv', index=False)
print("done!")
