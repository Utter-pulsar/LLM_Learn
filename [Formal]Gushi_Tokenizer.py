from transformers import BertTokenizer
from transformers import BertModel


tokenizer = BertTokenizer.from_pretrained("./models/Bert-Base-Chinese")
pretrained_model = BertModel.from_pretrained("./models/Bert-Base-Chinese")
tokens = tokenizer.encode("春眠不觉晓", max_length=12, padding = "max_length", truncation=True)
print(tokens)

print(tokenizer("春眠不觉晓", max_length=12, padding = "max_length", truncation=True))


