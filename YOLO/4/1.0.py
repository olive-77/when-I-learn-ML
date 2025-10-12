#SXP开头的是我自己写的注释，其他是豆包生成的
# 1. 导入必要库（固定）
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time 
#model_name = "uer/gpt2-chinese-base"  # 中文模型，而非默认"gpt2" SXP 镜像上面好像没有这个模型，其他的中文模型看半天都要登录，算了，就英文
model_name ='gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

device=('xpu')
model.to(device)
#model=torch.compile(model)
print("model加载成功")

with open(r'D:\My_CODE\Git_ML\when-I-learn-ML\YOLO\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
#len(text)

text=tokenizer(text,return_tensors='pt')

#生成标签
labels = text['input_ids'].detach().clone()
#SXP让标签与输入文本错开，左移一个位置   & SXP这里是最唐氏的错误：label与ids是相同的
#labels = labels.roll(-1, dims=1)
#labels[0, -1] = tokenizer.eos_token_id

print(" 数据加载成功")
  #  '''
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
for epoch in range(2):  #训练过程中其实可以换成小批次
    s=time.time()
    text = {k: v.to(device) for k, v in text.items()}
    labels=labels.to(device)
    outputs = model(**text, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    t=time.time()-s
    print(loss,t)

  #  '''
# 3. 模型交互核心逻辑（挖空版，需你补充）
def model_interact(model, tokenizer, user_input: str):
    """
    实现“用户输入→模型输出”的交互
    user_input: 你的提问（比如“鸣人，什么是忍道？”）
    """

    # TODO 1：构建提示词（参考格式：用户：xxx\n鸣人：）
    # 示例方向：拼接成符合角色对话的格式，引导模型生成对应风格回复
    prompt = f"Users:{user_input}\nKobe:" # 请补充，比如 f"用户：{user_input}\n鸣人："
# SXP1. f是python中的一种语法，诣在直接往引用句里面填入{变量名}
# SXP2. 这样写的目的是为了在每次交互的过程中，有一个动态的prompt。也就是说，这个函数是同时用在训练微调和模型测试上的
    # TODO 2：将提示词转为模型可识别的token（调用tokenizer）
    # 示例方向：用tokenizer.encode或tokenizer(return_tensors="pt")
    inputs= tokenizer(prompt,return_tensors="pt")
    inputs = inputs.to(device)  
    # 请补充，注意将inputs放到模型所在设备（model.device）
#SXP这样看来，这里还只是实际使用的时候的函数，并没有微调部分的空间。
#SXP不过问题不大，到时后写个判断语句，一种拿prommt，一种拿csv的数据。
#SXP哦，最好的是直接传进来的就是相同模式的，那么就之接
    # TODO 3：模型生成回复（调用model.generate）
    # 关键参数：max_new_tokens（生成长度）、do_sample（是否多样）、temperature（随机性）
    outputs = model.generate(
        **inputs,  # 传入inputs
        max_new_tokens=64,  # 建议填50-100
        do_sample=True,  # 建议True（多样）
        #SXP这里的do——sample就是说不一定选择概率最高的token，而是会有一定随机性
        #SXP通常与temperature绑定，温度越小越接近0，那么生成的东西越根据概率最大
        #SXP但超过了1的话就会比较混论。
        temperature=0.5,  # 建议0.7-0.9（平衡连贯与多样）  
        eos_token_id=[tokenizer.eos_token_id] , # 固定：遇到结束符停止
        pad_token_id = tokenizer.eos_token_id
    )#SXP这里生成的outputs的size为【B，T】，是一个ids的矩阵！

    # TODO 4：将模型输出的token解码为文字（调用tokenizer.decode）
    # 注意：用skip_special_tokens=True去掉无用符号
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # 请补充outputs相关参数

    # TODO 5：提取纯回复（去掉prompt部分，只保留模型生成的内容）
    # 示例方向：按prompt格式分割，取“鸣人：”后面的内容
    final_response =  raw_response.split("\n")[1]# 请补充，比如 raw_response.split("鸣人：")[-1]
    final_response =  final_response.replace("Kobe:", "").strip() #SXP调整输出格式
    return final_response




# 4. 主程序入口（固定，直接运行）
if __name__ == "__main__":  
#SXP这句是ai写的，是别的程序调用本程序的时候用的，也算长知识了。
#SXP懒得删掉
    # 第一步：下载加载模型  
    #SXP已经在最上面了
    if not model or not tokenizer:
        exit()  # 模型加载失败则退出
    
    # 第二步：循环交互（固定）
    print("\n=== 开始交互（输入'退出'结束）===")
    while True:
        user_text = input("You:")  #SXP这里有输入哦
        if user_text in ["退出", "q"]:
            print("模型：再见！")
            break
        # 调用交互逻辑（你补充的部分会在这里生效）
        reply = model_interact(model, tokenizer, user_text)
        print(f"Kobe Bryant:{reply}")