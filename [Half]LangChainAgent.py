from langchain.agents import tool, initialize_agent, AgentType
from typing import Optional, Union
from langchain.tools import BaseTool
from math import cos, sqrt, sin
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


model_type = ModelType.qwen1half_0_5b_chat
template_type = get_default_template_type(model_type)
model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'}, model_id_or_path = './models/Qwen1.5-1.8B-Chat')
device = "cuda"

pipe = pipeline(
    "text-generation",
    model = model,
    tokenizer=tokenizer,
    max_new_tokens = 128,
    temperature = 0.4
)

Qwen = HuggingFacePipeline(pipeline = pipe)


description = ("当你需要计算直角三角形的斜边长度时可使用此工具，"
               "给定直角三角形的一边或两边和/或一个角度（以度为单位）。"
               "使用此工具是，必须提供以下参数中的至少两个："
               "['adjacent_side', 'opposite_side', 'angle']。")

class HypotenuseTool(BaseTool):
    name = "Hypotenuse Calculator"
    description = description
    def _run(self,
             adjacent_side: Optional[Union[int, float]] = None,
             opposite_side: Optional[Union[int, float]] = None,
             angle: Optional[Union[int, float]] = None
             ):
        if adjacent_side and opposite_side:
            return sqrt(float(adjacent_side)**2 + float(opposite_side)**2)
        elif adjacent_side and angle:
            return adjacent_side/cos(float(angle))
        elif opposite_side and angle:
            return opposite_side/sin(float(angle))
        else:
            return "无法计算三角形斜边长度。需要提供更多参数。谢谢您嘞。"

tools = [HypotenuseTool()]

agent = initialize_agent(tools = tools,
                         llm = Qwen,
                         agent = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                         verbose = True,
                         max_iterations = 3,
                         handle_parsing_errors = True)
agent.run("如果有一个直角三角形，两直角边的长度分别是3厘米和4厘米，那么斜边的长度是多少？")
agent.run("如果有一个直角三角形，其中一个角是45度，对边长度为4厘米，那么斜边的长度是多少？")
agent.run("如果有一个直角三角形，其中一个角是45度，邻边长度是3厘米，那么斜边的长度是多少？")



