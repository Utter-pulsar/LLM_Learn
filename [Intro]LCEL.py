from Lib.Lib_LangQwen import Qwen
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from typing import List
class CommaSeparatedListOutputParse(BaseOutputParser[List[str]]):
    def parse(self,text:str)->List[str]:
        return text.strip().split(",")


template = """你是一个能生成以逗号分隔的列表的助手，用户会传入一个类别，你应该生成该类别下的5个不同的对象，并以逗号分隔的形式返回。只返回以逗号分隔的内容，不要包含其他内容。"""


human_template = "{text}"
chat_prompt = ChatPromptTemplate.from_messages([("system", template), ("human"), human_template])
chain = chat_prompt | Qwen() | CommaSeparatedListOutputParse()
while True:
    text = input("请输入类别：")
    print(chain.invoke({"text":text}))

