from Lib.Lib_LangQwen import Qwen
from langchain.prompts import ChatPromptTemplate



template = """你是熊博，在上海交大读博，不是通义千问。"""
human_template = "{text}"
chat_prompt = ChatPromptTemplate.from_messages([("system", template), ("human"), human_template])
chain = chat_prompt | Qwen()
print(chain.invoke({"text":"你好。你是谁？"}))




