import jieba.posseg as pseg
import csv
import random
from tqdm import tqdm
import hanzipy
def changeVerbPosition(text):
    phr = pseg.cut(text)
    positionMoveRange = 2
    phrases = []
    words = []
    verbs = []
    verbs_index = []
    cnt = 0
    b = 0
    for i, j in phr:
        cnt+=1
        if j == 'v':
            verbs.append(i)
            verbs_index.append(cnt)
            continue
        phrases.append(j)
        words.append(i)
    changed_verbs_index = []
    for i in verbs_index:
        new_index = min(max(i + random.randrange(-positionMoveRange, positionMoveRange), 0),cnt)
        count = 0
        while changed_verbs_index.count(new_index) != 0:
            count += 1
            if (count == 10000):
                changed_verbs_index = verbs_index
                break
            new_index = max(i + random.randrange(-positionMoveRange, positionMoveRange), 0)
        changed_verbs_index.append(new_index)
    for i in range(len(verbs)):
        words.insert(changed_verbs_index[i], verbs[i])
    return ''.join(words)
def SplitAndChangeVerbPosition(text, method="verb"):
    symbles = ["。", "？", "，", "！", "…", "?", "!", ".", "{", "}", ",",]
    finalText = []
    waitingText=''
    for i in text:
        if i in symbles:
            if method == "verb":
                finalText.append(changeVerbPosition(waitingText))
            waitingText = ''
            finalText.append(i)
        else :
            waitingText+=i
    if waitingText != '':
        finalText.append(changeVerbPosition(waitingText))
    # print(finalText)
    return ''.join(finalText)
# how_many_for_each = 3
# with open("augmented.tsv", "r") as tsv_file:
#     with open("change_verb.tsv", "w") as tsv_result:
#         tsv_reader = csv.reader(tsv_file, delimiter="\t")
#         tsv_writer = csv.writer(tsv_result, delimiter="\t")
#         tsv_writer.writerow(["Original Scentence", "After Change Verb Position"])
#         next(tsv_reader)
#         for line in tqdm(tsv_reader):
#             for i in range(how_many_for_each):
#                 after_change = splitSentence(line[0])
#                 if after_change == line[0]:
#                     tsv_writer.writerow(["", "No Differences After Change"])
#                     continue
#                 tsv_writer.writerow(["", after_change])