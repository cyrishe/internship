import pandas as pd

df_m3 = df_mr = df_qr = None
df = []
name = []

def readexcels():
    global df_m3, df_mr, df_qr, df, name
    df_m3 = pd.read_excel("C:\\Users\\schoology\\Desktop\\shanghai\\1ssst\\Stage_2\\m3_tsv_results_enhanced_mix.xlsx")
    df_mr = pd.read_excel("C:\\Users\\schoology\\Desktop\\shanghai\\1ssst\\Stage_2\\bger_tsv_results_mix.xlsx")
    df_qr = pd.read_excel("C:\\Users\\schoology\\Desktop\\shanghai\\1ssst\\Stage_2\\qwen_tsv_results_mix.xlsx")

    df = [df_m3,df_mr,df_qr]
    name = ['m3','bger','qwen']

readexcels()
for i in range(3):
    similarity = df[i]['similarity']
    label = df[i]['label']

    range_1 = 0
    cnt1 = 0
    cnt0 = 0
    range_0 = 0
    # print(len(similarity))
    # print(cnt1)
    for j in range(len(similarity)):
        if label[j]:
            range_1 += similarity[j]
            cnt1 += 1
        else:
            range_0 += similarity[j]
            cnt0 += 1
    #不保证两个cnt都大于1，为0时输出0
    dif_in_1 = range_1 / cnt1 if cnt1 > 0 else 0
    dif_in_0 = range_0 / cnt0 if cnt0 > 0 else 0

    print(f"{name[i]}: 正例平均相似度 = {dif_in_1}, 负例平均相似度 = {dif_in_0}，差值：{abs(dif_in_1 - dif_in_0)}")
    print(f"{name[i]}: 正例数量 = {cnt1}, 负例数量 = {cnt0}, ")