import torch
import torch.nn as nn
from torch.nn import functional as F
#import matplotlib.pyplot as plt
import numpy as np
import intel_extension_for_pytorch as ipex
import time 
## 超参设置（每个参数都有讲究！）
batch_size = 16     # 批处理大小
block_size = 32     # 上下文长度
max_iters = 5000    # 训练轮数
learning_rate = 1e-3 # 学习率
device ='xpu'
n_embd = 64         # embedding维度
n_head = 4          # 多头注意力头数
n_layer = 4         # transformer层数
dropout = 0.0       # dropout率
eval_interval=50
torch.manual_seed(42) # 随机种子，保证可复现

# 读取莎士比亚文本
with open('D:/My_CODE/Git_ML/when-I-learn-ML/微光/8 transformers/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"数据集总字符数: {len(text):,}")
print(f"前300个字符预览:\n{text[:300]}")

# 构建词汇表
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"词汇表: {''.join(chars)}")
print(f"词汇表大小: {vocab_size}")

# 简单tokenizer实现
stoi = {ch: i for i, ch in enumerate(chars)}   #encoder
itos = {i: ch for i, ch in enumerate(chars)}   #decoder
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 测试编解码
print(f"编码测试: {encode('Hello World!')}")
print(f"解码测试: {decode(encode('Hello World!'))}")

text1=text[:int(0.8*len(text))]
text2=text[int(0.8*len(text)):]
text1=encode(text1)
text2=encode(text2)
def get_batch(split):
    """
    生成训练批次数据
    
    你需要实现：
    1. 从训练/验证集中随机采样  ok
    2. 生成输入序列x和目标序列y    ok  no 这里随机打乱是不对的，这样会产生乱码
    3. 返回正确维度的tensor  
    """
    ###################
    if split=='train':
        data=text1
    else :
        data= text2
    data = torch.tensor(data)
    idx=torch.randint(high=len(data)-block_size-1,size=(batch_size,))
    inputs=torch.stack([data[i:i+block_size] for i in idx ])
    outputs=torch.stack([data[i+1:i+block_size+1]  for i in idx])
    return inputs.to(device),outputs.to(device)
    ###################
    pass
class Head(nn.Module):
    """单头自注意力"""
    
    def __init__(self, head_size):  #传来embedding长度   这里不是一开始的n_embd啊，多头会把这个砍掉的
        super().__init__()
        # QKV投影矩阵
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False) 
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # 下三角mask矩阵（防止看到未来）
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        ###################
        # 注意力计算逻辑
        # 1. 计算Q, K, V               
        Q = self.key(x)
        K = self.query(x)
        V = self.value(x)
        # 2. 计算注意力分数  有个scale   
        score = Q@ K.transpose(-1,-2)/(C ** 1/2)
        # 3. 应用mask 这里有softmax的
        score=score.masked_fill(self.tril==0,float('-inf'))
        score=torch.softmax(score,dim=-1)
        # 4. 加权求和                
        ans=score@V
        return ans
        ###################
        pass

class MultiHeadAttention(nn.Module):
    """多头注意力：并行处理多个注意力头"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        ###################
        # 多头并行计算与合并
        x= [head(x) for head in self.heads]
        x=torch.cat(x,dim=-1)
        x=self.proj(x)
        x=self.dropout(x)
        return x
        ###################
        pass

class FeedForward(nn.Module):   
    """前馈神经网络：简单而强大"""
    ###################             
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear(n_embd,n_embd) 
        self.relu=nn.ReLU()
        self.l2=nn.Linear(n_embd,n_embd)
    def forward (self,x):
        x=self.l1(x)
        x=self.relu(x)
        x=self.l2(x)
        return x
    ###################
    pass

class Block(nn.Module):  #不需要考虑循环
    """Transformer Block：communication + computation"""
    ###################
    # 组装完整的transformer块
    # LayerNorm + MultiHeadAttention + LayerNorm + FeedForward
    def __init__(self,n_embd,n_head): 
        super().__init__()
        self.l1=nn.LayerNorm(n_embd)
        self.muL=MultiHeadAttention(num_heads=n_head,head_size=n_embd//n_head)
        self.l2=nn.LayerNorm(n_embd)
        self.FFN=FeedForward()
    def forward(self,x):
        x=self.l1(x)
        x=self.muL(x)
        x=self.l2(x)
        x=self.FFN(x)
        return x
    ###################
    pass

class BigramLanguageModel(nn.Module):
    """完整的语言模型"""
    
    def __init__(self):
        super().__init__()
        # embedding表
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # 堆叠transformer块
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        ###################
        # 完整前向传播
        # 1. token + position embedding
        token= self.token_embedding_table(idx)
        pos=self.position_embedding_table(torch.arange(idx.size(1), device=idx.device))
        out=token+pos
        # 2. 通过transformer块
        out=self.blocks(out)
        out=self.ln_f(out)
        out=self.lm_head(out)
        # 3. 输出预测    这一点真的小脑萎缩了
        logits = out
        print("shape:", logits.shape,targets.shape)
        # 4. 计算损失（如果有target）  训练的时候必然是有target的，因为有返回loss
        if targets==None:
            return logits
        else :
            loss=F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),reduction='mean')
            print("loss shape:", loss.shape)
            return logits,loss
        ###################
        pass
    
    def generate(self, idx, max_new_tokens):
        """文本生成：让AI说话！"""
        for _ in range(max_new_tokens):
            # 裁剪到block_size
            idx_cond = idx[:, -block_size:]
            # 预测下一个token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # 只要最后一个时间步
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
@torch.no_grad()
def estimate_loss():   ##有点没必要了吧，你下面都没调用
    """评估训练和验证损失"""
    out = {}
    model.eval()
    ###################
    # 计算平均损失
    ###################
    model.train()
    return out
# 实例化模型
model = BigramLanguageModel().to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_losses, val_losses = [], []
print("开始训练")
for iter in range(max_iters):
    ###################             #这里真的蚌埠住了。没有主代码我是真的写不下去了，数据规格完全是一脸懵逼啊
    # 训练循环实现
    t1=time.time()
    # 1. 获取batch
    inputs ,targets = get_batch('train') 
    # 2. 前向传播
    logits, loss = model(inputs , targets)  
    # 3. 计算损失
    train_losses.append(loss.item())
    # 4. 反向传播   
    loss.backward()
    # 5. 参数更新   
    optimizer.step()
    optimizer.zero_grad()
    torch.xpu.synchronize()
    t2=time.time()
    t=t2-t1
    print(iter ,'次迭代',t)
    # 6. 定期评估
    if iter % eval_interval == 0:
        print('laoda')
        with torch.no_grad():
            print('maba')
            val_logits, val_loss = model(*get_batch('val'))   #你这样写了那前面那个评估函数是真用不上了  #*解包元组
            print('out')
            val_losses.append(val_loss.item())
    ###################
    pass
'''
# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='#FF6B6B')
plt.plot(val_losses, label='Validation Loss', color='#4ECDC4')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Progress: Loss Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')

'''
# 生成莎士比亚风格文本
print("AI莎士比亚开始创作...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_new_tokens=2000)[0].tolist())
print(generated_text)