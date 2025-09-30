import torch
import torch.nn as nn
from numpy.distutils.fcompiler import intel
from torch.nn import functional as F
import intel_extension_for_pytorch as intel
# 仅保留必要超参数，batch_size将动态测试
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = 'xpu'  # 根据实际设备调整

# 数据准备（简化版，仅用于获取词汇表大小）
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)


# 模型定义（保持与原模型结构一致）
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss


# 测试最大批次大小的函数
def find_max_batch_size():
    model = GPTLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model, optimizer = intel.optimize(model, optimizer=optimizer, device=device)
    # 从较小的批次开始测试
    batch_size = 1
    max_possible = 1

    while True:
        try:
            # 创建测试数据
            x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
            y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)

            # 前向传播+反向传播（模拟训练过程）
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # 测试成功，记录当前批次并尝试更大批次
            max_possible = batch_size
            batch_size *= 2  # 指数增长加速测试
            print(f"成功运行批次大小: {max_possible}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"内存不足，最大可行批次大小为: {max_possible}")
                # 清理内存
                torch.cuda.empty_cache()
                return max_possible
            else:
                print(f"其他错误: {e}")
                return max_possible
        except Exception as e:
            print(f"测试失败: {e}")
            return max_possible


# 执行测试
if __name__ == "__main__":
    max_batch = find_max_batch_size()
    print(f"最终确定的最大批次大小: {max_batch}")