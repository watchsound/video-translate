import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

prompt = """
in the following content, some sentences may be splitted into several lines.
please reformat the  content, such that one whole sentence should only occupy one line


我們來介紹一下子如何
處理這個數據公司輸入
其實數據公司輸入的聽起來簡單是一個挑戰的活
卻有非常有用
比如說一兩年前
好未來在總結他們當年的產品取得的功能裡面
特地強調了
完成了數據公司輸入
我們來看一下這你是如何之前假設我們
真有各種方式你給簡單輸入用的貸責
提供不如有螢幕
然後我們真有看你一個USR判斷
他都一樣沒有年後貸就是直接是例子貸就是這個
這個IPP這個存訊你了
其實沒什麼用
但是很好玩吧
假設我是寫出來一個一個
二 加三 更好 四吧
這個因為是用了這個
都曾經往落他用
加完寫的數據速度特別慢
沒有多大實際用處
你看他還盤有點盤斷錯了
因為差不多就是這樣

"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that understand language."},
        {"role": "user", "content": prompt}
    ],
  #  max_tokens=150,
    n=1,
    stop=None,
    temperature=0.9,
)
translation = response.choices[0].message.content.strip()
print(translation)