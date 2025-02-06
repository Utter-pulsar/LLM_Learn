from langchain.prompts import ChatPromptTemplate
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.tuners import Swift
ckpt_dir = './FineTuneModels/qwen1half-0_5b-chat/v0-20240621-140351/checkpoint-100'
model_type = ModelType.qwen1half_0_5b_chat
template_type = get_default_template_type(model_type)
model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'}, model_id_or_path = './Qwen1.5-0.5B-Chat')
model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
model.generation_config.max_new_tokens = 128

from abc import ABC
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
device = "cuda" # the device to load the model onto


class Qwen(LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    history_len: int = 3

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "Qwen"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"max_token": self.max_token,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "history_len": self.history_len}



template = """你是一个炸鸡店的客服，不是通义千问。"""
human_template = "{text}"
chat_prompt = ChatPromptTemplate.from_messages([("system", template), ("human"), human_template])
chain = chat_prompt | Qwen()



if __name__ == "__main__":
    print(chain.invoke({"text":"你好。你是谁？"}))
