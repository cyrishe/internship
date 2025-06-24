import csv

with open("result_wrong_writing.tsv") as file_csv:
    reader = csv.reader(file_csv, delimiter="\t")
    with open("result_wrong_writing_utf.tsv", "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for i in reader:
            writer.writerow(i)
