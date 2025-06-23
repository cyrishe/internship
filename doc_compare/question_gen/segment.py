import os
import json

directory_path = "./jiantou_doc_text"
txt_files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
final = {}
final_files = {}
for file_name in txt_files:
    full_file_path = os.path.join(directory_path, file_name)
    with open(full_file_path) as file:
        final_files[file_name] = file.read()
length = 0
for i in final_files:
    father_file_name = i
    segments = [s for s in final_files[i].split("\n\n\n") if s.strip()]
    length += len(segments)
    final[i] = segments
print(length)
print(final["【AI百宝箱】怎样从众多线索中挖掘投资机会？.txt"][0])
with open("segments.json", "w") as json_file:
    json.dump(final, json_file)
