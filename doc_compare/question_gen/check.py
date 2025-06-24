import csv
from openai import OpenAI
import threading
import queue
import json
import ast

threads = []
stop_event = threading.Event()
result_queue = queue.Queue()

key = "sk-c6b7fef3a032484caec5b9cf6db96a9d"

client = OpenAI(
    api_key=key,
    base_url="https://api.deepseek.com",
)

with open("check_prompt.json") as json_file:
    systemPrompt = json_file.read()


def askDeepSeek(text) -> dict:
    messages = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": text},
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


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


def under_ques(line: list):
    question_list = ast.literal_eval(line[2])
    final_question_list = []
    for ques in question_list["question"]:
        askQues = ques.copy()
        askQues["text"] = line[1]
        ans = askDeepSeek(str(askQues))
        result: bool = ans["result"]
        reference: str = ans["text"]
        ques["check"] = result
        ques["reference"] = reference
        final_question_list.append(ques)
    question_list["question"] = final_question_list[:]
    result_queue.put([line[0], line[1], question_list])


cnt = -1
from tqdm import tqdm

with open("result.csv") as file_csv:
    reader = csv.reader(file_csv, delimiter="\t")
    next(reader)
    reader = list(reader)
    writer = threading.Thread(target=writer_thread, args=("check_result.tsv",))
    writer.start()
    for line in tqdm(reader, total=len(list(reader))):
        # print(cnt)
        if cnt == 0:
            break
        cnt -= 1
        # print(f"line: {ast.literal_eval(line[2])}")
        t = threading.Thread(target=under_ques, args=(line,))
        threads.append(t)
        t.start()
    for t in tqdm(threads, total=len(threads)):
        t.join()
    stop_event.set()
    writer.join()
