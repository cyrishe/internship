import os
import torch
import json
from modelscope import AutoModelForCausalLM, AutoTokenizer

# ====================== 1. 环境配置 ======================

# 避免显存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 tokenizer 和 model（用 FP16 减少显存占用）
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-Reranker-0.6B",
    torch_dtype=torch.float16
).to(device).eval()

# ====================== 2. Prompt 配置 ======================

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

max_length = 2048  # 减小最大长度节省显存
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

# ====================== 3. 构造输入 & 推理函数 ======================

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

def process_inputs(text):
    inputs = tokenizer(
        text,
        padding=False,
        truncation='longest_first',
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    inputs['input_ids'] = [prefix_tokens + inputs['input_ids'] + suffix_tokens]
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    return inputs

@torch.no_grad()
def compute_score(inputs):
    logits = model(**inputs).logits[:, -1, :]
    yes_logits = logits[:, token_true_id]
    no_logits = logits[:, token_false_id]
    log_probs = torch.nn.functional.log_softmax(torch.stack([no_logits, yes_logits], dim=1), dim=1)
    return log_probs[:, 1].exp().item()  # yes的概率

# ====================== 4. 加载数据 & 小段处理 ======================

# 加载文档数据
with open('jiantou_doc_text_split.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

query = "你的查询内容"
task = 'Given a web search query, retrieve relevant passages that answer the query'

# 按小段（每条doc）逐个处理
all_scores = []
for idx, doc in enumerate(documents):
    try:
        # 1. 构造输入
        prompt = format_instruction(task, query, doc)

        # 2. 编码输入并推理
        inputs = process_inputs(prompt)
        score = compute_score(inputs)

        all_scores.append((idx, score))
        print(f"[{idx}] Score: {score:.4f}")

        # 3. 显存清理
        del inputs
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        print(f"[{idx}] ❌ CUDA OOM，跳过此条")
        torch.cuda.empty_cache()
        all_scores.append((idx, None))
