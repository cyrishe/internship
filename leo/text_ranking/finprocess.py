import torch
from modelscope import AutoModel, AutoTokenizer, AutoModelForCausalLM
from FlagEmbedding import BGEM3FlagModel, FlagReranker
import pandas as pd
import gc
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

df = pd.read_csv('C:\\Users\\schoology\\Desktop\\shanghai\\1ssst\\Stage_2\\test_data.tsv', sep='\t', encoding='utf-8')

sentences_1 = df['测试问题'].tolist()
sentences_2 = df['原始问题'].tolist()
wrong_se = [[] for _ in range(5)]

answers = df['答案'].tolist()
for i in range(5):
    for j in range(len(sentences_1)):
        wrong_se[i].append(sentences_2[(j + i * 20 + len(sentences_1)) % len(sentences_1)])
    df[f'测试数据{i+1}'] = wrong_se[i]

# ---------------------- bge-m3 ----------------------
model = BGEM3FlagModel('C:\\Users\\schoology\\Desktop\\shanghai\\1ssst\\models\\models\\bge-m3', use_fp16=True)

for i in range(5):
    sentences_2 = wrong_se[i]
    embeddings_1 = model.encode(sentences_1, batch_size=12, max_length=2048)['dense_vecs']
    embeddings_2 = model.encode(sentences_2)['dense_vecs']

    answers = []
    for j in range(len(embeddings_1)):
        answers.append(embeddings_1[j] @ embeddings_2[j].T)

    df[f'm3测试数据{i+1}结果'] = answers
    torch.cuda.empty_cache()
    gc.collect()

# ---------------------- bge-reranker ----------------------
reranker = FlagReranker('models/bge-reranker', use_fp16=True)

for i in range(5):
    sentences_2 = wrong_se[i]
    pairs = list(zip(sentences_1, sentences_2))
    try:
        label_br = reranker.compute_score(pairs, normalize=True)
    except ValueError as e:
        print(f"[Error] bge-reranker 第{i+1}组处理失败：{e}")
        label_br = [0] * len(pairs)
    df[f'bgerank测试数据{i+1}结果'] = label_br
    torch.cuda.empty_cache()
    gc.collect()

# ---------------------- qwen3-reranker ----------------------
def format_instruction(instruction, query, doc):
    instruction = instruction or 'Given a web search query, retrieve relevant passages that answer the query'
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    return batch_scores[:, 1].exp().tolist()

# tokenizer + model
tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-Reranker-06B", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("models/Qwen3-Reranker-06B", torch_dtype=torch.float16).cuda().eval()
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 2048
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
task = 'Given a web search query, retrieve relevant passages that answer the query'

# doing qwen3 reranker 5 times so that each time it uses a different set of wrong sentences
for i in range(5):
    sentences_2 = wrong_se[i]
    pairs = [format_instruction(task, q, d) for q, d in zip(sentences_1, sentences_2)]

    label_qwen = []
    batch_size = 8
    try:
        for start in range(0, len(pairs), batch_size):
            sub_pairs = pairs[start:start + batch_size]
            inputs = process_inputs(sub_pairs)
            label_qwen.extend(compute_logits(inputs))
            torch.cuda.empty_cache()
            gc.collect()
    except torch.cuda.OutOfMemoryError:
        print(f"[OOM] qwen3 第{i+1}组显存不足，打0分")
        label_qwen = [0] * len(sentences_1)

    df[f'qwen测试问题{i+1}答案'] = label_qwen
    torch.cuda.empty_cache()
    gc.collect()

df.to_excel("final.xlsx", index=False)
print('done!')
