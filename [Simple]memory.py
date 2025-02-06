from Lib.Lib_AgentQwen import Qwen

from langchain.memory import ConversationEntityMemory
llm = Qwen
memory = ConversationEntityMemory(llm = llm)

_input = {"input":"小李和莫尔索正在参加一场AI领域的黑客马拉松。"}
memory.load_memory_variables(_input)
memory.save_context(_input, {"output":"听起来真不错，他们在做什么项目？"})
print(memory.load_memory_variables({"input":"莫尔索在干嘛？"}))





