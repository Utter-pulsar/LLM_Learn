from typing import List
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from Lib.Lib_LangQwen import Qwen
from langchain.schema import BaseOutputParser
from langserve import add_routes

class CommaSeparatedListOutputParse(BaseOutputParser[List[str]]):
    def parse(self,text:str)->List[str]:
        return text.strip().split(",")


template = """你是一个能生成以逗号分隔的列表的助手，用户会传入一个类别，你应该生成该类别下的5个不同的对象，并以逗号分隔的形式返回。只返回以逗号分隔的内容，不要包含其他内容。"""


human_template = "{text}"
chat_prompt = ChatPromptTemplate.from_messages([("system", template), ("human"), human_template])
chain = chat_prompt | Qwen() | CommaSeparatedListOutputParse()

app = FastAPI(title = "应用", version = "0.0.1", description = "Langchain接口",)

add_routes(app, chain, path = "/firstapp")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port = 8000)
    # 可以在http://localhost:8000/firstapp/playground/上面看到界面

