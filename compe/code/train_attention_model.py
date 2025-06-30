import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# 参数设置
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
EPOCHS = 5
BATCH_SIZE = 32

# === Attention Layer === #
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform",
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, return_attention=False):
        e = K.tanh(K.dot(x, self.W))           # (batch, time_steps, 1)
        a = K.softmax(e, axis=1)               # 注意力权重 (batch, time_steps, 1)
        context = x * a                        # 加权后的表示
        context = K.sum(context, axis=1)       # 汇总为一个向量

        if return_attention:
            return context, a
        return context

# === 读取和准备数据 === #
df = pd.read_csv("train.csv")
df.fillna("", inplace=True)
df["full_text"] = df["keyword"] + " " + df["location"] + " " + df["text"]
texts = df["full_text"].values
labels = df["target"].values

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y = labels

# 训练验证集划分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === 构建模型（训练模型） === #
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
embedding = Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM)(input_)
lstm_out = LSTM(64, return_sequences=True)(embedding)

# 用 Attention 层输出 context vector
att_layer = AttentionLayer()
context = att_layer(lstm_out)

x = Dropout(0.5)(context)
x = Dense(32, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=input_, outputs=output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# === 训练 === #
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE)

model.save("tweet_attention_model_v2.h5")

# === 构建 Attention 权重输出模型 === #
# Attention 层重新调用，带 return_attention=True
att_layer_extract = AttentionLayer()
context2, att_weights = att_layer_extract(lstm_out, return_attention=True)
att_model = Model(inputs=input_, outputs=[att_weights, output])

# === 从验证集中提取关键词（修复 padding 问题）=== #
word_index = tokenizer.word_index
reverse_word_index = {v: k for k, v in word_index.items()}
keyword_counter = {}
total_extracted = 0

for i in range(min(200, len(X_val))):
    sample_input = X_val[i:i+1]
    att_w, pred = att_model.predict(sample_input, verbose=0)
    weights = att_w[0].squeeze()  # 长度应为 MAX_SEQUENCE_LENGTH

    token_ids = sample_input[0]
    tokens = [reverse_word_index.get(tid, "<PAD>") for tid in token_ids]

    # 打印一条样本 attention 状态（只执行一次）
    if i == 0:
        print("\n📌 第一个样本的 Token 和 Attention:")
        for tok, weight in zip(tokens, weights):
            print(f"{tok:>15}: {weight:.4f}")

    # 获取 top3 高权重索引（不跳 PAD）
    top_indices = weights.argsort()[-3:][::-1]

    for idx in top_indices:
        word = tokens[idx]
        if word != "<PAD>":
            keyword_counter[word] = keyword_counter.get(word, 0) + 1
            total_extracted += 1

print(f"\n✅ 共提取关键词数（非去重）：{total_extracted}")

# 统计前50关键词
sorted_keywords = sorted(keyword_counter.items(), key=lambda x: -x[1])[:50]
if sorted_keywords:
    with open("attention_keywords.json", "w") as f:
        json.dump(sorted_keywords, f, indent=2)
    print("\n✅ attention_keywords.json 已保存")
else:
    print("\n⚠️ 没有有效关键词被提取，可能是模型没学会或者输入太短")

