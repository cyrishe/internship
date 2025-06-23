import json
from openai import OpenAI
import math, collections
import csv
from tqdm import tqdm
import threading

threads = []


def shannon_entropy_cn(sentence: str) -> float:
    counter = collections.Counter(sentence)
    total = sum(counter.values())
    return -sum((freq / total) * math.log2(freq / total) for freq in counter.values())


key = "sk-c6b7fef3a032484caec5b9cf6db96a9d"

client = OpenAI(
    api_key=key,
    base_url="https://api.deepseek.com",
)
with open("segments.json") as json_file:
    question = json.load(json_file)
with open("prompt.json") as f:

    system_prompt = f.read()
    system_prompt = system_prompt.split("---------split, iginore me!!!!")[0]


def askDeepSeek(text):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


import queue

# 使用队列实现线程安全的数据传递
result_queue = queue.Queue()
threads = []
stop_event = threading.Event()


def under_j(j, file_name):
    try:

        content = j
        prev = []
        numberOfQuestion = max(int(2 ** shannon_entropy_cn(content) / 16 + 0.5), 3)
        for i in range(numberOfQuestion):
            nowUserPrompt = {"content": content, "prev": prev}
            # print(nowUserPrompt)
            nowUserPrompt = json.dumps(nowUserPrompt)
            ans = askDeepSeek(nowUserPrompt)
            prev.append(ans)
        finalJson = {"number": len(prev), "question": prev}
        # 将结果放入队列
        result_queue.put(
            (
                file_name.replace("\n", "\\n"),
                content.replace("\n", "\\n"),
                str(finalJson).replace("\n", "\\n"),
            )
        )
    except Exception as e:
        print(f"线程错误: {e}")


def writer_thread(csv_file_path):
    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        writer.writerow(["father_file", "question", "answer"])

        while not (stop_event.is_set() and result_queue.empty()):
            try:
                # 从队列获取结果，设置超时避免永久阻塞
                row = result_queue.get(timeout=0.1)
                writer.writerow(row)
                result_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"写入错误: {e}")


def main():
    global stop_event

    # 先启动写入线程
    writer_t = threading.Thread(target=writer_thread, args=("result.csv",))
    writer_t.start()

    cnt = -1

    try:
        for i in tqdm(question, total=len(question)):
            file_name = i
            for j in question[i]:
                if cnt == 0:
                    break
                t = threading.Thread(target=under_j, args=(j, file_name))
                threads.append(t)
                t.start()
                cnt -= 1

            if cnt == 0:
                break

        # 等待所有工作线程完成
        for t in tqdm(threads, total=len(threads)):
            t.join()

    finally:
        # 通知写入线程可以退出了
        stop_event.set()
        # 等待写入线程完成
        writer_t.join()
        print("所有线程执行完毕")


main()
