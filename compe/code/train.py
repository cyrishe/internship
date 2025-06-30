import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle

# ======================
# 🔧 配置参数
# ======================
MAX_WORDS = 20000
MAX_LEN_TEXT = 80
EMBEDDING_DIM = 128
EPOCHS = 50
BATCH_SIZE = 64

# ======================
# 🧩 读取训练数据
# ======================
df = pd.read_csv('train.csv')
df['location'] = df['location'].fillna('')
df['keyword'] = df['keyword'].fillna('')
df['text'] = df['text'].fillna('')
df['combined_text'] = df['text'] + ' ' + df['keyword'] + ' ' + df['location']
labels = df['target'].values

# ======================
# 🔠 文本预处理
# ======================
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['combined_text'])
sequences = tokenizer.texts_to_sequences(df['combined_text'])
padded_text = pad_sequences(sequences, maxlen=MAX_LEN_TEXT, padding='post', truncating='post')

# ======================
# 📊 数据划分
# ======================
X_train, X_val, y_train, y_val = train_test_split(padded_text, labels, test_size=0.2, random_state=42)

# ======================
# 🧠 构建模型
# ======================
input_text = Input(shape=(MAX_LEN_TEXT,), name='text_input')
x = Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM)(input_text)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dropout(0.3)(x)
x = Bidirectional(LSTM(64))(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_text, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ======================
# 🛑 回调函数设置
# ======================
early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# ======================
# 🚀 训练模型
# ======================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    # callbacks=[early_stop, checkpoint, lr_scheduler]
)

# ======================
# 💾 保存模型和处理器
# ======================
model.save('twitter_model_final.h5')
with open('tokenizer_combined.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("✅ 模型和 tokenizer 已保存。")

# ======================
# 📈 可视化训练过程
# ======================
plt.figure(figsize=(12, 5))

# Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves_improved.png')
print("📊 已保存训练曲线图 training_curves_improved.png")

# ======================
# 🧾 保存训练日志
# ======================
pd.DataFrame(history.history).to_csv('training_log.csv', index=False)
print("📄 已保存训练日志 training_log.csv")
