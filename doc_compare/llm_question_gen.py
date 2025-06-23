import json
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
import threading
import queue
client = OpenAI(
    api_key="sk-c6b7fef3a032484caec5b9cf6db96a9d",
    base_url="https://api.deepseek.com",
)

system_prompt = """
The user will provide some exam text. Please generate sone "question-answer" pairs and output them in JSON format in Chinese. It is wonderful if you control the amount of pairs in shannon value. 

EXAMPLE INPUT: 
| 北上资金策略要素表（相关数据更新至2023.07.31） | Unnamed: 1 | Unnamed: 2 | Unnamed: 3 |
|-|-|-|-|
| 基本信息 | 7 | 三大签约理由 | 1、2018-2023年2月，策略收益110%，最大回撤31%，夏普比率0.74（回测数据仅代表过去，不代表未来表现） |
| |              | | 2、股票调入调出都有明确显示，有明确的风控止损线，控制风险。 |
| |              | | 3、中长线操作，降低交易费用，减少出错频率，组合投资吃个股大波段。 |

EXAMPLE JSON OUTPUT:
{
  'shennon_entropy': 3.5,
  {
    "question": "北上资金策略的回测收益情况如何？",
    "answer": "2018年至2023年2月期间，该策略收益为110%，最大回撤为31%，夏普比率为0.74。"
  },
  {
    "question": "该策略是如何控制风险的？",
    "answer": "策略明确显示股票的调入调出，并设有明确的风控止损线来控制风险。"
  },
  {
    "question": "该策略的操作周期和投资方式是怎样的？",
    "answer": "该策略采用中长线操作，降低交易费用，减少出错频率，并通过组合投资捕捉个股大波段。"
  }
}

"""

with open("jiantou_doc_text_split.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# print(data)

answers = []
length = len(data)
print(type(data))
writingQueue = queue.Queue()
tread = []
is_done = threading.Thread()
user_prompt = None

def multipro(user_prompt):
  global system_prompt, client, answers
  messages = [{"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt}]

  response = client.chat.completions.create(
      model="deepseek-chat",
      messages=messages,
      response_format={
        'type': 'json_object'
      }
    )
  answers.append(response.choices[0].message.content)

for i in tqdm(range(length)):
  #  multipro(data[i]) 
  t = threading.Thread(target=multipro, args=(data[i],))
  tread.append(t)
  t.start()
for i in tqdm(tread, total=len(tread)):
  i.join()
with open("processed_data.json", "w", encoding="utf-8") as f:
    json.dump(answers, f, ensure_ascii=False, indent=2)


df = pd.DataFrame({"原始数据":data , "生成问题": answers})

df.to_excel("output.xlsx", index=False)

print("done!")
