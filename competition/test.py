from transformers import TrainingArguments

args = TrainingArguments(
    output_dir='./test',
    evaluation_strategy="epoch",   # 这个报错说明版本太老
    num_train_epochs=1
)

print("✅ transformers 版本支持 TrainingArguments 参数")
