from template import CoT_consistent_refine_factors_prompt_chinese, materials
from chatGPT import get_GPT_response
import json
import re
origin_template = CoT_consistent_refine_factors_prompt_chinese

question = input("请输入您的问题：")
prompt = origin_template.format(question=question, materials=materials)
print("=====输入的问题=====")
print(prompt)
result = get_GPT_response(model="gpt-4o",message=prompt)
print("=====输出的回答=====")
print(result)
## 在回答中寻找json字符串.```json```标记的内容
json_str = re.search(r'```json(.*?)```', result, re.S).group(1)
print("=====json字符串=====")
print(json_str)
## 将json字符串转换为字典
json_dict = json.loads(json_str)
print("=====json字典=====")
print(json_dict)
## 保存json字典到文件
## 每次先将上一次的文件，转移至备份文件backup文件夹下，如果没有则创建，同时重命名为当前的时间
import os
import shutil
import time
if not os.path.exists("backup"):
    os.mkdir("backup")
shutil.move("factors.json", "backup/factors_"+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())+".json")
with open('factors.json', 'w',encoding='utf-8') as f:
    json.dump(json_dict, f, ensure_ascii=False, indent=4)
print("=====保存到文件成功=====")
