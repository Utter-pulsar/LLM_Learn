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
    temperature = 0.1
)

Qwen = HuggingFacePipeline(pipeline = pipe)