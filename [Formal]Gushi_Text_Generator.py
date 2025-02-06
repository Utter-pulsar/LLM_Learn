from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("./models/gpt2-chinese-poem")
model = GPT2LMHeadModel.from_pretrained("./models/gpt2-chinese-poem")
text_generator = TextGenerationPipeline(model, tokenizer)
# result = text_generator("[CLS]万叠春山积雨晴，", max_length = 50, do_sample = True)
result = text_generator("[CLS]鹅鹅鹅，", max_length = 100, do_sample = True)


print(result)











