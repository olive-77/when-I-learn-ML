import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')  #以上代码可以直接从网页上面的官方介绍里面copy
tokenizer.pad_token=tokenizer.eos_token #eos是句子末尾的符号，pad是batch之间不等长时的填充符号

with open(r'D:\My_CODE\Git_ML\when-I-learn-ML\YOLO\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
#len(text)
text=tokenizer(text,return_tensors='pt')
#生成标签
labels = text['input_ids'].detach().clone()
#让标签与输入文本错开，左移一个位置
labels = labels.roll(-1, dims=1)
labels[0, -1] = tokenizer.eos_token_id


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
for epoch in range(3):
    outputs = model(**text, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
