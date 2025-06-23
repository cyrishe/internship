import pandas as pd

df = pd.read_csv("result.csv", delimiter="\t")
df.to_excel("result.xlsx")
