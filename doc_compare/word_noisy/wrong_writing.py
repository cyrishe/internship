import json
import Levenshtein
import random
from tqdm import tqdm
import change_verb
char_table = {}
with open('homo_and_sim.json', 'r') as wrong_writing_table:
    char_table = json.load(wrong_writing_table)
second_char_table = {}
with open('sim.json', 'r') as wrong_writing_table:
    second_char_table = json.load(wrong_writing_table)
def levenshtein_similarity(str1, str2):
    dist = Levenshtein.distance(str1, str2)
    max_len = max(len(str1), len(str2))
    return 1 - dist / max_len
from pycorrector import Corrector
m = Corrector()
def changeSentence(sentence: str, chance: float = .7, accessable: float = 0.75, maxium_try: int = 10, secondary_amount: float = 0.02):
    result = ''
    cnt = maxium_try
    while levenshtein_similarity(m.correct(result)['target'], sentence) < accessable:
        result = ''
        if cnt<=1:
            return f"{sentence}"
        for i in sentence:
            canChange = random.randrange(0, 100) / 100.0
            if (i in char_table) and (canChange <= chance):
                result += random.choice(char_table[i])
            elif (i in second_char_table) and ((canChange) <= (chance * secondary_amount)):
                result += random.choice(second_char_table[i])
            else:
                result += i
        cnt-=1
    return result
import csv
import queue
import threading
result_queue = queue.Queue()
threads = []
stop_event = threading.Event()
def under_i(fileName, segment, ques):
    global nowThread
    result_queue.put((
                fileName,
                segment,
                ques,
                change_verb.SplitAndChangeVerbPosition(changeSentence(ques)), 
                change_verb.SplitAndChangeVerbPosition(changeSentence(ques)), 
                change_verb.SplitAndChangeVerbPosition(changeSentence(ques)),
                change_verb.SplitAndChangeVerbPosition(changeSentence(ques)), 
                change_verb.SplitAndChangeVerbPosition(changeSentence(ques)), 
    ))
def writer_thread(csv_file_path):
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        writer.writerow(['Original File Name','Original Segment','Original Sentence', 'Wrong Writing Sentence 1', 'Wrong Writing Sentence 2', 'Wrong Writing Sentence 3'])
        while not (stop_event.is_set() and result_queue.empty()):
            try:
                row = result_queue.get(timeout=0.1)
                writer.writerow(row)
                result_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(e)
with open('result.csv', 'r') as tsv_file:
    tsv_reader = csv.reader(tsv_file,delimiter='\t')
    next(tsv_reader)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    tsv_reader = list(tsv_reader)
    writer_thread_at_main = threading.Thread(target=writer_thread, args=("result_wrong_writing.tsv",))
    writer_thread_at_main.start()
    try:
        for i in tqdm(tsv_reader, total=len(tsv_reader)):
            while nowThread>maxiuxThread:
                for t in tqdm(threads, total=len(threads)):
                    t.join()
                    threads.remove(t)
            ques = i[2]
            fileName = i[0]
            segment = i[1]
            t = threading.Thread(target=under_i,args=(fileName, segment, ques))
            
            threads.append(t)

            t.start()
            nowThread+=1
        for t in tqdm(threads, total=len(threads)):
            t.join()
    finally:
        stop_event.set()
        writer_thread_at_main.join()
        print(f'things all done')