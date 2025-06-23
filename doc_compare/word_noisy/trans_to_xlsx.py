import pandas as pd

df = pd.read_csv("result_wrong_writing.tsv", delimiter="\t")
df.to_excel("result_wrong_writing.xlsx")
