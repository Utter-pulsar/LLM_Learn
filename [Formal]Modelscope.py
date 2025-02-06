import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)

model_type = ModelType.qwen1half_0_5b_chat
template_type = get_default_template_type(model_type)
model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'}, model_id_or_path = './Qwen1.5-0.5B-Chat')
model.generation_config.max_new_tokens = 128
template = get_template(template_type, tokenizer)
query = ('问题的模板是：帮我设计一个{}边形的图案。'
         '用户输入的问题是：帮我设计一个六边形的图案。'
         '请把用户输入的问题按照模板当中的格式提取出括号中的参数，只需要回复括号内的参数数字即可，其他一概不用回答。')

response, history = inference(model, template, query)
print(f'response: {response}')
print(f'history: {history}')