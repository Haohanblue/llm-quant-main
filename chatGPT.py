from dotenv import load_dotenv
import os
import json
from openai import OpenAI
# 加载 .env 文件
load_dotenv(verbose=True)
client = OpenAI(
# This is the default and can be omitted
base_url='https://api.gptsapi.net/v1',
api_key=os.environ.get("OPENAI_API_KEY"),
)
def get_GPT_response(message,model):
  
    response = client.chat.completions.create(
        model=model,  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": message},
        ],
        # 拓展参数
        extra_body={"temperature": 1},
        )
    return response.choices[0].message.content

if __name__ == '__main__':
    print(get_GPT_response(model="gpt-4o",message="你好"))