import torch
import pandas as pd
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 0. 检查 CUDA 可用性并设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device} (CUDA: {torch.cuda.is_available()})")
va_o = 100

# 1. 加载数据
try:
    typeout = int(input("请输入数据类型（1: 测试 vs 原始, 2: 测试 vs 原始+答案）: "))
    df = pd.read_csv('test_data.tsv', sep='\t')
    queries = df['测试问题'].tolist()
    documents = df['原始问题'].tolist()
    answers = df['答案'].tolist()
    doc = documents
    ans = answers
    print(documents[1])
    for i in range(len(documents)):
        documents[i] = doc[(i + va_o) % len(documents)] # 偏移制造反例
    print(documents[1])
    for i in range(len(answers)):
        answers[i] = ans[(i + va_o) % len(answers)]  # 偏移制造反例
    df['答案'] = answers  # 更新 DataFrame 中的答案列
    df['原始问题'] = documents  # 更新 DataFrame 中的原始问题列
    if typeout == 2:
        mix = [f"{doc}\n{ans}" for doc, ans in zip(documents, answers)]
    # mix = [f"{doc}\n{ans}" for doc, ans in zip(documents, answers)]
except Exception as e:
    raise ValueError(f"数据加载失败: {e}")

# 2. 加载模型（自动适配设备）
try:
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-Reranker-06B", padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        "models/Qwen3-Reranker-06B",
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32  # GPU用半精度节省显存
    ).eval().to(device)
except Exception as e:
    raise RuntimeError(f"模型加载失败: {e}")

# 3. 定义常量
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 2048  # 安全长度
threshold = 0.82
batch_size = 4 if device == 'cuda' else 2  # GPU用更大批次

# 4. 格式化函数
def format_instruction(query, doc, instruction=None):
    instruction = instruction or 'Given a web search query, retrieve relevant passages that answer the query'
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

# 5. 批量处理函数
def process_batch(batch_pairs):
    inputs = tokenizer(
        batch_pairs,
        padding=True,
        truncation='longest_first',
        return_tensors="pt",
        max_length=max_length
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        scores = torch.softmax(
            torch.stack([logits[:, token_false_id], logits[:, token_true_id]], dim=1),
            dim=1
        )[:, 1].tolist()
    return scores

# 6. 主处理流程
try:
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    pairs = [format_instruction(task, query, documents) for query, documents in zip(queries, mix)] if typeout == 2 else documents
    
    similarity_scores = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="计算相似度", unit="batch"):
        batch = pairs[i:i + batch_size]
        similarity_scores.extend(process_batch(batch))
        if device == 'cuda':
            torch.cuda.empty_cache()

    # 7. 保存结果
    df['similarity'] = similarity_scores
    df['label'] = (df['similarity'] > threshold).astype(int)
    df.to_excel('qwen_similarity_results_mix_negative.xlsx', index=False) if typeout == 2 else df.to_excel('similarity_results_negative.xlsx', index=False)
    print(f"处理完成！结果已保存到 qwen_similarity_results_mix_negative.xlsx") if typeout == 2 else print(f"处理完成！结果已保存到 similarity_results_negative.xlsx")

except Exception as e:
    print(f"处理过程中出错: {e}")
    if 'CUDA out of memory' in str(e):
        print("提示: 尝试减小 batch_size 或 max_length")