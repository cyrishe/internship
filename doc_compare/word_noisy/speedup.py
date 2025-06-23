from concurrent.futures import ThreadPoolExecutor
import queue
import threading
# 增加写入线程数量
WRITER_THREADS = 16

def writer_thread(csv_file_path, queue):
    with open(csv_file_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        writer.writerow([...])
        while True:
            try:
                row = queue.get(timeout=1)
                if row is None:  # 终止信号
                    break
                writer.writerow(row)
            except queue.Empty:
                continue

# 主程序
result_queue = queue.Queue()
writer_threads = []

# 启动多个写入线程
for _ in range(WRITER_THREADS):
    t = threading.Thread(target=writer_thread, args=("result.tsv", result_queue))
    t.start()
    writer_threads.append(t)

# 使用线程池处理任务
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    for row in tsv_reader:
        futures.append(executor.submit(under_i, row[0], row[1], row[2]))
    
    for future in tqdm(futures):
        future.result()  # 等待完成

# 发送终止信号
for _ in range(WRITER_THREADS):
    result_queue.put(None)
    
for t in writer_threads:
    t.join()