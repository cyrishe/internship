import os
import json

directory = "C:\\Users\\schoology\\Desktop\\shanghai\\jiantou_doc_text\\jiantou_doc_text"

files = []

for file in os.listdir(directory):
    if not file.startswith("._"):
        files.append(os.path.join(directory, file))

# print(files)
# for file in files:
#     with open(file, "r", encoding="utf-8") as f:
#         content = f.read()
#         if "\n\n\n" in content:
#             print(f"{file} 有三个连续换行")
#         else:
#             print(f"{file} 没有三个连续换行")

contents = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        parts = content.split("\n\n\n")
        # print(parts)
        contents.extend(parts)

contents = [part for part in contents if part.strip()]


with open("jiantou_doc_text_split.json", "w", encoding="utf-8") as f:
    json.dump(contents, f, ensure_ascii=False, indent=4)
